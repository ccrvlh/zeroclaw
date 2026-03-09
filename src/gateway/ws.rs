//! WebSocket agent chat handler.
//!
//! Protocol:
//! ```text
//! Client -> Server: {"type":"message","content":"Hello"}
//! Server -> Client: {"type":"chunk","content":"Hi! "}
//! Server -> Client: {"type":"tool_call","name":"shell","args":{...}}
//! Server -> Client: {"type":"tool_result","name":"shell","output":"..."}
//! Server -> Client: {"type":"done","full_response":"..."}
//! ```

use super::AppState;
use crate::providers::ChatRequest;
use axum::{
    extract::{
        ws::{Message, WebSocket},
        Query, State, WebSocketUpgrade,
    },
    response::IntoResponse,
};
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use std::time::Instant;

#[derive(Deserialize)]
pub struct WsQuery {
    pub token: Option<String>,
}

/// GET /ws/chat — WebSocket upgrade for agent chat
pub async fn handle_ws_chat(
    State(state): State<AppState>,
    Query(params): Query<WsQuery>,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    // Auth via query param (browser WebSocket limitation)
    if state.pairing.require_pairing() {
        let token = params.token.as_deref().unwrap_or("");
        if !state.pairing.is_authenticated(token) {
            return (
                axum::http::StatusCode::UNAUTHORIZED,
                "Unauthorized — provide ?token=<bearer_token>",
            )
                .into_response();
        }
    }

    ws.on_upgrade(move |socket| handle_socket(socket, state))
        .into_response()
}

async fn handle_socket(socket: WebSocket, state: AppState) {
    let (mut sender, mut receiver) = socket.split();

    while let Some(msg) = receiver.next().await {
        let msg = match msg {
            Ok(Message::Text(text)) => text,
            Ok(Message::Close(_)) => break,
            Err(_) => break,
            _ => continue,
        };

        // Parse incoming message
        let parsed: serde_json::Value = match serde_json::from_str(&msg) {
            Ok(v) => v,
            Err(_) => {
                let err = serde_json::json!({"type": "error", "message": "Invalid JSON"});
                let _ = sender.send(Message::Text(err.to_string().into())).await;
                continue;
            }
        };

        let msg_type = parsed["type"].as_str().unwrap_or("");
        if msg_type != "message" {
            continue;
        }

        let content = parsed["content"].as_str().unwrap_or("").to_string();
        if content.is_empty() {
            continue;
        }

        // Process message with the LLM provider
        let provider_label = state
            .config
            .lock()
            .default_provider
            .clone()
            .unwrap_or_else(|| "unknown".to_string());
        let (provider, model, temperature) = state.llm_snapshot();
        let started_at = Instant::now();

        state
            .observer
            .record_event(&crate::observability::ObserverEvent::AgentStart {
                provider: provider_label.clone(),
                model: model.clone(),
            });
        state
            .observer
            .record_event(&crate::observability::ObserverEvent::LlmRequest {
                provider: provider_label.clone(),
                model: model.clone(),
                messages_count: 1,
            });

        // Broadcast agent_start event
        let _ = state.event_tx.send(serde_json::json!({
            "type": "agent_start",
            "provider": provider_label.clone(),
            "model": model.clone(),
        }));

        // Simple single-turn chat (no streaming for now — use provider.chat_with_system)
        let system_prompt = {
            let config_guard = state.config.lock();
            crate::channels::build_system_prompt(
                &config_guard.workspace_dir,
                &model,
                &[],
                &[],
                Some(&config_guard.identity),
                None,
            )
        };

        let messages = vec![
            crate::providers::ChatMessage::system(system_prompt),
            crate::providers::ChatMessage::user(&content),
        ];

        let multimodal_config = state.config.lock().multimodal.clone();
        let prepared =
            match crate::multimodal::prepare_messages_for_provider(&messages, &multimodal_config)
                .await
            {
                Ok(p) => p,
                Err(e) => {
                    let err = serde_json::json!({
                        "type": "error",
                        "message": format!("Multimodal prep failed: {e}")
                    });
                    let _ = sender.send(Message::Text(err.to_string().into())).await;
                    continue;
                }
            };

        match provider
            .chat(
                ChatRequest {
                    messages: &prepared.messages,
                    tools: None,
                },
                &model,
                temperature,
            )
            .await
        {
            Ok(response) => {
                let duration = started_at.elapsed();
                let (input_tokens, output_tokens) = response
                    .usage
                    .as_ref()
                    .map(|u| (u.input_tokens, u.output_tokens))
                    .unwrap_or((None, None));
                state
                    .observer
                    .record_event(&crate::observability::ObserverEvent::LlmResponse {
                        provider: provider_label.clone(),
                        model: model.clone(),
                        duration,
                        success: true,
                        error_message: None,
                        input_tokens,
                        output_tokens,
                    });
                state.observer.record_metric(
                    &crate::observability::traits::ObserverMetric::RequestLatency(duration),
                );
                state
                    .observer
                    .record_event(&crate::observability::ObserverEvent::AgentEnd {
                        provider: provider_label.clone(),
                        model: model.clone(),
                        duration,
                        tokens_used: input_tokens.zip(output_tokens).map(|(i, o)| i + o),
                        cost_usd: None,
                    });

                // Send the full response as a done message
                let done = serde_json::json!({
                    "type": "done",
                    "full_response": response.text_or_empty(),
                });
                let _ = sender.send(Message::Text(done.to_string().into())).await;

                // Broadcast agent_end event
                let _ = state.event_tx.send(serde_json::json!({
                    "type": "agent_end",
                    "provider": provider_label,
                    "model": model,
                }));
            }
            Err(e) => {
                let duration = started_at.elapsed();
                let sanitized = crate::providers::sanitize_api_error(&e.to_string());
                state
                    .observer
                    .record_event(&crate::observability::ObserverEvent::LlmResponse {
                        provider: provider_label.clone(),
                        model: model.clone(),
                        duration,
                        success: false,
                        error_message: Some(sanitized.clone()),
                        input_tokens: None,
                        output_tokens: None,
                    });
                state.observer.record_metric(
                    &crate::observability::traits::ObserverMetric::RequestLatency(duration),
                );
                state
                    .observer
                    .record_event(&crate::observability::ObserverEvent::Error {
                        component: "ws_chat".to_string(),
                        message: sanitized.clone(),
                    });
                state
                    .observer
                    .record_event(&crate::observability::ObserverEvent::AgentEnd {
                        provider: provider_label.clone(),
                        model: model.clone(),
                        duration,
                        tokens_used: None,
                        cost_usd: None,
                    });

                let err = serde_json::json!({
                    "type": "error",
                    "message": sanitized,
                });
                let _ = sender.send(Message::Text(err.to_string().into())).await;

                // Broadcast error event
                let _ = state.event_tx.send(serde_json::json!({
                    "type": "error",
                    "component": "ws_chat",
                    "message": sanitized,
                }));
            }
        }
    }
}
