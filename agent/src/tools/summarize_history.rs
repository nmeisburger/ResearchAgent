use crate::Result;
use crate::llm::{CompletionRequest, LLM, Message};
use crate::tools::{Tool, ToolCall, ToolDefinition};
use async_trait::async_trait;
use std::sync::Arc;

pub struct SummarizeHistory {
    llm: Arc<dyn LLM + Send + Sync>,
    keep_last: usize,
}

impl SummarizeHistory {
    pub fn new(llm: Arc<dyn LLM + Send + Sync>, keep_last: usize) -> Box<Self> {
        Box::new(Self { llm, keep_last })
    }

    pub async fn summarize_history(&self, mut messages: Vec<Message>) -> Result<Vec<Message>> {
        // assume that the first two messages are the system and user prompt which contains the task instructions
        if messages.len() < 2 + self.keep_last {
            return Ok(messages);
        }

        let last_messages = messages.split_off(messages.len() - self.keep_last);

        messages.push(Message::User(PROMPT.to_string()));

        let result = self
            .llm
            .completion(CompletionRequest {
                messages: &messages,
                tools: &vec![],
            })
            .await?;

        let _ = messages.split_off(2);
        messages.push(Message::Assistant(result.content, vec![]));
        messages.extend(last_messages.into_iter());

        Ok(messages)
    }
}

const PROMPT: &str = "In order to keep the conversational history from becoming to long, you must generate a summary of the current chat history. 
Instructions: 
- The summary must compress the information, try to be as succinct as possible. The finaly summary should not be more than 1000 words in length.
- Preserve key information from the conversational history. Remember that information stored using the memory tool can still be retrieved later. 
- Remember that you are a researcher, make sure to preserve any key findings or information that you will need to complete the task.";

#[async_trait]
impl Tool for SummarizeHistory {
    fn definition(&self) -> Result<ToolDefinition> {
        return Ok(ToolDefinition::new::<()>(
            "summarize_history",
            &format!(
                "This tool will take in the chat history, and generate a concise summary that preserves the key component. This prevents the conversational history from becoming too long, and makes it easier to find the relevant information in the history. Note that the last {} messages will not be changed, only the preceding messages will be summarized. Remember that you should also use the memory tool to store key information for retrieval later. You must use this tool to prevent the history from becoming too long. It will automatically be invoked if the chat history becomes too long.",
                self.keep_last
            ),
        )?);
    }

    async fn invoke(&mut self, _: &ToolCall, messages: Vec<Message>) -> Result<Vec<Message>> {
        self.summarize_history(messages).await
    }
}
