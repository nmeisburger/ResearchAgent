use crate::Result;
use crate::llm::Message;
use crate::tools::SummarizeHistory;
use async_trait::async_trait;

mod logger;
pub use logger::MessageLogger;

#[async_trait]
pub trait Callback {
    async fn call(&mut self, messages: Vec<Message>) -> Result<Vec<Message>>;
}

#[async_trait]
impl Callback for SummarizeHistory {
    async fn call(&mut self, messages: Vec<Message>) -> Result<Vec<Message>> {
        if messages.iter().map(Message::ntokens).sum::<usize>() > 5000 {
            return self.summarize_history(messages).await;
        }
        Ok(messages)
    }
}
