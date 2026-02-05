#!/usr/bin/env python3
"""
Convert conversation data to pretraining format for GPT-OSS training.
In pretraining mode, all tokens are unmasked (no distinction between user/assistant).
"""

import json
from pathlib import Path

import typer

app = typer.Typer()


def convert_conversation_to_pretrain(sample: dict) -> dict:
    """Convert a conversation sample to pretraining format."""

    # Collect all message content directly (no prefixes)
    text_parts = []

    if "messages" in sample:
        for msg in sample["messages"]:
            thinking = msg.get("thinking", "")
            content = msg.get("content", "").strip()

            # Add thinking first if present
            if thinking and thinking.strip():
                text_parts.append(thinking.strip())

            # Then add main content
            if content:
                text_parts.append(content)

    # Join all parts with double newlines for separation
    full_text = "\n\n".join(text_parts)

    # Create pretraining format - just raw text, no chat template
    pretrain_sample = {"messages": [{"role": "pretrain", "content": full_text}]}

    return pretrain_sample


@app.command()
def convert(
    input_file: str = typer.Option(..., "--input-file", help="Input JSONL file path"),
    output_file: str = typer.Option(..., "--output-file", help="Output JSONL file path"),
):
    """Convert conversation data to pretraining format."""

    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        typer.echo(f"Error: Input file {input_path} does not exist")
        raise typer.Exit(1)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converted_count = 0
    error_count = 0

    typer.echo(f"Converting {input_path} to pretraining format...")

    with (
        open(input_path, encoding="utf-8") as infile,
        open(output_path, "w", encoding="utf-8") as outfile,
    ):
        for line_num, line in enumerate(infile, 1):
            try:
                # Parse JSON line
                sample = json.loads(line.strip())

                # Convert to pretraining format
                pretrain_sample = convert_conversation_to_pretrain(sample)

                # Skip if no content
                if not pretrain_sample["messages"][0]["content"].strip():
                    error_count += 1
                    continue

                # Write converted sample
                json.dump(pretrain_sample, outfile, ensure_ascii=False)
                outfile.write("\n")

                converted_count += 1

                if converted_count % 1000 == 0:
                    typer.echo(f"Processed {converted_count} samples...")

            except Exception as e:
                typer.echo(f"Error processing line {line_num}: {e}")
                error_count += 1
                continue

    typer.echo("✅ Conversion complete!")
    typer.echo(f"📊 Converted: {converted_count} samples")
    typer.echo(f"❌ Errors: {error_count} samples")
    typer.echo(f"💾 Output saved to: {output_path}")

    # Show sample of converted data
    typer.echo("\n📝 Sample converted data:")
    with open(output_path, encoding="utf-8") as f:
        first_line = f.readline()
        sample_data = json.loads(first_line)
        content_preview = sample_data["messages"][0]["content"][:300] + "..."
        typer.echo(f"Content preview: {content_preview}")


if __name__ == "__main__":
    app()
