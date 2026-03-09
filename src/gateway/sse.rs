//! Server-Sent Events (SSE) stream for real-time event delivery.
//!
//! Wraps the broadcast channel in AppState to deliver events to web dashboard clients.

use super::AppState;
use axum::{
    extract::State,
    http::{header, HeaderMap, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse,
    },
};
use std::convert::Infallible;
use std::sync::Arc;
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;

/// GET /api/events — SSE event stream
pub async fn handle_sse_events(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> impl IntoResponse {
    // Auth check
    if state.pairing.require_pairing() {
        let token = headers
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|auth| auth.strip_prefix("Bearer "))
            .unwrap_or("");

        if !state.pairing.is_authenticated(token) {
            return (
                StatusCode::UNAUTHORIZED,
                "Unauthorized — provide Authorization: Bearer <token>",
            )
                .into_response();
        }
    }

    let rx = state.event_tx.subscribe();
    let stream = BroadcastStream::new(rx).filter_map(
        |result: Result<
            serde_json::Value,
            tokio_stream::wrappers::errors::BroadcastStreamRecvError,
        >| {
            match result {
                Ok(value) => Some(Ok::<_, Infallible>(
                    Event::default().data(value.to_string()),
                )),
                Err(_) => None, // Skip lagged messages
            }
        },
    );

    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// Broadcast observer that forwards events to the SSE broadcast channel.
pub struct BroadcastObserver {
    inner: Box<dyn crate::observability::Observer>,
    tx: tokio::sync::broadcast::Sender<serde_json::Value>,
    cost_tracker: Option<Arc<crate::cost::CostTracker>>,
}

impl BroadcastObserver {
    pub fn new(
        inner: Box<dyn crate::observability::Observer>,
        tx: tokio::sync::broadcast::Sender<serde_json::Value>,
        cost_tracker: Option<Arc<crate::cost::CostTracker>>,
    ) -> Self {
        Self {
            inner,
            tx,
            cost_tracker,
        }
    }
}

impl crate::observability::Observer for BroadcastObserver {
    fn record_event(&self, event: &crate::observability::ObserverEvent) {
        // Forward to inner observer
        self.inner.record_event(event);

        match event {
            crate::observability::ObserverEvent::LlmResponse {
                success: true,
                model,
                input_tokens,
                output_tokens,
                ..
            } => {
                tracing::debug!(
                    model = %model,
                    input_tokens = ?input_tokens,
                    output_tokens = ?output_tokens,
                    has_cost_tracker = self.cost_tracker.is_some(),
                    "Received successful LLM response event for cost tracking"
                );

                if let Some(cost_tracker) = &self.cost_tracker {
                    if let Err(error) =
                        cost_tracker.record_llm_usage(model, *input_tokens, *output_tokens)
                    {
                        tracing::warn!("Failed to record LLM usage for cost tracking: {error}");
                    }
                } else {
                    tracing::warn!(
                        "Cost tracker is not configured; LLM usage cannot be persisted for dashboard cost"
                    );
                }
            }
            _ => {}
        }

        // Broadcast to SSE subscribers
        let json = match event {
            crate::observability::ObserverEvent::LlmRequest {
                provider, model, ..
            } => serde_json::json!({
                "type": "llm_request",
                "provider": provider,
                "model": model,
                "timestamp": chrono::Utc::now().to_rfc3339(),
            }),
            crate::observability::ObserverEvent::ToolCall {
                tool,
                duration,
                success,
            } => serde_json::json!({
                "type": "tool_call",
                "tool": tool,
                "duration_ms": duration.as_millis(),
                "success": success,
                "timestamp": chrono::Utc::now().to_rfc3339(),
            }),
            crate::observability::ObserverEvent::ToolCallStart { tool } => serde_json::json!({
                "type": "tool_call_start",
                "tool": tool,
                "timestamp": chrono::Utc::now().to_rfc3339(),
            }),
            crate::observability::ObserverEvent::Error { component, message } => {
                serde_json::json!({
                    "type": "error",
                    "component": component,
                    "message": message,
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                })
            }
            crate::observability::ObserverEvent::AgentStart { provider, model } => {
                serde_json::json!({
                    "type": "agent_start",
                    "provider": provider,
                    "model": model,
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                })
            }
            crate::observability::ObserverEvent::AgentEnd {
                provider,
                model,
                duration,
                tokens_used,
                cost_usd,
            } => serde_json::json!({
                "type": "agent_end",
                "provider": provider,
                "model": model,
                "duration_ms": duration.as_millis(),
                "tokens_used": tokens_used,
                "cost_usd": cost_usd,
                "timestamp": chrono::Utc::now().to_rfc3339(),
            }),
            _ => return, // Skip events we don't broadcast
        };

        let _ = self.tx.send(json);
    }

    fn record_metric(&self, metric: &crate::observability::traits::ObserverMetric) {
        self.inner.record_metric(metric);
    }

    fn flush(&self) {
        self.inner.flush();
    }

    fn name(&self) -> &str {
        "broadcast"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::schema::{CostConfig, ModelPricing};
    use crate::observability::traits::ObserverMetric;
    use crate::observability::{Observer, ObserverEvent};
    use parking_lot::Mutex;
    use std::collections::HashMap;
    use std::time::Duration;
    use tempfile::TempDir;

    #[derive(Default)]
    struct TestObserver {
        events: Mutex<u64>,
    }

    impl Observer for TestObserver {
        fn record_event(&self, _event: &ObserverEvent) {
            *self.events.lock() += 1;
        }

        fn record_metric(&self, _metric: &ObserverMetric) {}

        fn name(&self) -> &str {
            "test-observer"
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    #[test]
    fn llm_response_event_records_cost_usage() {
        let tmp = TempDir::new().unwrap();
        let mut prices = HashMap::new();
        prices.insert(
            "test/model".to_string(),
            ModelPricing {
                input: 1.0,
                output: 2.0,
            },
        );
        let cost_tracker = Arc::new(
            crate::cost::CostTracker::new(
                CostConfig {
                    enabled: true,
                    prices,
                    ..Default::default()
                },
                tmp.path(),
            )
            .unwrap(),
        );

        let (tx, _rx) = tokio::sync::broadcast::channel(8);
        let observer = BroadcastObserver::new(
            Box::new(TestObserver::default()),
            tx,
            Some(cost_tracker.clone()),
        );
        observer.record_event(&ObserverEvent::LlmResponse {
            provider: "openrouter".to_string(),
            model: "test/model".to_string(),
            duration: Duration::from_millis(10),
            success: true,
            error_message: None,
            input_tokens: Some(1000),
            output_tokens: Some(500),
        });

        let summary = cost_tracker.get_summary().unwrap();
        assert_eq!(summary.request_count, 1);
        assert!(summary.session_cost_usd > 0.0);
    }
}
