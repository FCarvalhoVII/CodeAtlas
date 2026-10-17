from pathlib import Path

from codeatlas.indexing.scanner import RepositoryScanner
from codeatlas.parsing.service import RepositoryPreparer


def test_repository_to_chunks_never_executes_source(tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    source = (
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
        'raise RuntimeError("never execute")\n\ndef run():\n    return 1\n'
    )
    (tmp_path / "main.py").write_text(source, encoding="utf-8")
    (tmp_path / "broken.py").write_text("def broken(:", encoding="utf-8")
    (tmp_path / "README.md").write_text("Documentation", encoding="utf-8")
    scan = RepositoryScanner().scan(tmp_path)
    result = RepositoryPreparer().prepare(scan)
    assert not marker.exists()
    assert {s.file_path for s in result.skipped} == {"broken.py", "README.md"}
    assert result.chunks
    for chunk in result.chunks:
        ref = chunk.reference
        lines = (tmp_path / ref.file_path).read_text(encoding="utf-8").splitlines(keepends=True)
        assert chunk.content == "".join(lines[ref.start_line - 1 : ref.end_line])
