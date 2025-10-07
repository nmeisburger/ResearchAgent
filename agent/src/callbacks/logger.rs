use crate::Result;
use crate::callbacks::Callback;
use crate::llm::Message;
use async_trait::async_trait;
use std::io::Write;

pub struct MessageLogger<W: Write + Send> {
    last_hashes: Vec<u64>,
    writer: W,
    step: u32,
}

impl<W: Write + Send> MessageLogger<W> {
    pub fn new(name: &str, mut writer: W) -> Result<Box<Self>> {
        write!(writer, "## {}\n\n", name)?;

        Ok(Box::new(Self {
            last_hashes: Vec::new(),
            writer: writer,
            step: 0,
        }))
    }

    fn display_messages(&mut self, messages: &[Message]) -> Result<()> {
        write!(self.writer, "### Step {}\n", self.step)?;

        messages
            .iter()
            .try_for_each(|m| write!(self.writer, "{}", m))?;

        write!(self.writer, "---\n")?;

        Ok(())
    }

    fn display_history_cleared(&mut self) -> Result<()> {
        write!(self.writer, "## [HISTORY CLEARED]\n\n")?;
        Ok(())
    }

    fn prefix_match_len(&self, new_hashes: &[u64]) -> usize {
        new_hashes
            .iter()
            .zip(self.last_hashes.iter())
            .filter(|&(a, b)| *a == *b)
            .count()
    }
}

#[async_trait]
impl<W: Write + Send> Callback for MessageLogger<W> {
    async fn call(&mut self, messages: Vec<Message>) -> Result<Vec<Message>> {
        let new_hashes = messages.iter().map(Message::get_hash).collect::<Vec<_>>();

        if new_hashes.len() < self.last_hashes.len()
            || self.prefix_match_len(&new_hashes) != self.last_hashes.len()
        {
            self.display_history_cleared()?;
            self.display_messages(&messages)?;
        } else {
            self.display_messages(&messages[self.last_hashes.len()..])?;
        }

        self.writer.flush()?;

        self.step += 1;
        self.last_hashes = new_hashes;

        Ok(messages)
    }
}
