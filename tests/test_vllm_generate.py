import json

from gpu_slack_runner.jobs.vllm_generate import _read_prompts


def test_read_prompts_reads_all_jsonl_records(tmp_path) -> None:
    path = tmp_path / "prompts.jsonl"
    records = [{"prompt": "first"}, {"prompt": "second", "source": "test"}]
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    assert _read_prompts(path) == ["first", "second"]
