mod research;
use agent::Result;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The research task
    #[arg(short, long)]
    task: String,

    /// Name of the model to use
    #[arg(short, long)]
    model: String,

    /// Directory to store logs in
    #[arg(short, long, default_value = "./agent_logs")]
    log_dir: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let llm = agent::llm::OpenAI::new(args.model);

    let orchestrator =
        research::Orchestrator::new(llm, args.task, &std::path::Path::new(&args.log_dir))?;

    orchestrator.run().await?;

    Ok(())
}
