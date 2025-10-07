use crate::Result;
use crate::tools::{ToolCall, ToolDefinition};
use async_trait::async_trait;
use std::hash::{Hash, Hasher};

mod openai;
pub use openai::OpenAI;

#[derive(Clone, std::hash::Hash)]
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

    pub fn get_hash(&self) -> u64 {
        let mut hasher = std::hash::DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

impl std::fmt::Display for Message {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Message::Assistant(content, tool_calls) => {
                write!(f, "__Assistant:__ {}\n", content)?;
                tool_calls.iter().try_for_each(|t| ToolCall::fmt(t, f))?;
            }
            Message::System(content) => write!(f, "__System:__ {}\n", content)?,
            Message::User(content) => write!(f, "__User:__ {}\n", content)?,
            Message::Tool { id, name, result } => {
                write!(f, "__Tool:__ {} ({})\n{}\n", name, id, result)?
            }
        }

        f.write_fmt(format_args!("\n"))
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
