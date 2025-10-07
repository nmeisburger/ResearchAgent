use crate::Result;
use crate::tools::{ToolCall, ToolDefinition};
use async_trait::async_trait;

mod openai;
pub use openai::OpenAI;

#[derive(Clone)]
pub enum Message {
    User(String),
    Assistant(String, Vec<ToolCall>),
    System(String),
    Tool {
        id: String,
        name: String,
        result: String,
    },
}

impl Message {
    pub fn ntokens(&self) -> usize {
        match self {
            Message::User(content) => content.split_whitespace().count(),
            Message::Assistant(content, _) => content.split_whitespace().count(),
            Message::System(content) => content.split_whitespace().count(),
            Message::Tool { result, .. } => result.split_whitespace().count(),
        }
    }
}

pub struct CompletionRequest<'a> {
    pub messages: &'a [Message],
    pub tools: &'a [ToolDefinition],
    pub web_search_tool: bool,
}

pub struct CompletionResponse {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
}

#[async_trait]
pub trait LLM {
    async fn completion<'a>(&self, request: CompletionRequest<'a>) -> Result<CompletionResponse>;
}
