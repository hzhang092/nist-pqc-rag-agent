param(
    [ValidateSet(
        "help",
        "build",
        "ingest",
        "clean",
        "chunk",
        "embed",
        "index-faiss",
        "index-bm25",
        "search",
        "ask",
        "eval",
        "test",
        "pipeline"
    )]
    [string]$Task = "help",
    [string]$Query = "ML-KEM key generation"
)

$ErrorActionPreference = "Stop"

function Invoke-Compose([string[]]$ComposeArgs) {
    Write-Host "docker compose $($ComposeArgs -join ' ')" -ForegroundColor Cyan
    & docker compose @ComposeArgs
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose failed with exit code $LASTEXITCODE"
    }
}

function Invoke-Rag([string[]]$PyArgs) {
    $composeArgs = @("run", "--rm", "rag") + $PyArgs
    Invoke-Compose $composeArgs
}

function Show-Usage() {
    Write-Host "Usage:"
    Write-Host "  ./scripts/docker.ps1 -Task build"
    Write-Host "  ./scripts/docker.ps1 -Task pipeline"
    Write-Host "  ./scripts/docker.ps1 -Task search -Query 'Algorithm 19 ML-KEM.KeyGen'"
    Write-Host "  ./scripts/docker.ps1 -Task ask -Query 'What is ML-KEM?'"
}

switch ($Task) {
    "help" {
        Show-Usage
    }
    "build" {
        Invoke-Compose @("build", "rag")
    }
    "ingest" {
        Invoke-Rag @("python", "-m", "rag.ingest")
    }
    "clean" {
        Invoke-Rag @("python", "scripts/clean_pages.py")
    }
    "chunk" {
        Invoke-Rag @("python", "scripts/make_chunks.py")
    }
    "embed" {
        Invoke-Rag @("python", "-m", "rag.embed")
    }
    "index-faiss" {
        Invoke-Rag @("python", "-m", "rag.index_faiss")
    }
    "index-bm25" {
        Invoke-Rag @("python", "-m", "rag.index_bm25")
    }
    "search" {
        Invoke-Rag @("python", "-m", "rag.search", $Query)
    }
    "ask" {
        Invoke-Rag @("python", "-m", "rag.ask", $Query)
    }
    "eval" {
        Invoke-Rag @("python", "-m", "eval.run")
    }
    "test" {
        Invoke-Rag @(
            "python",
            "-m",
            "pytest",
            "tests/test_embed_store_records.py",
            "tests/test_retrieve_determinism.py",
            "tests/test_retrieve_rrf.py"
        )
    }
    "pipeline" {
        Invoke-Compose @("build", "rag")
        Invoke-Rag @("python", "-m", "rag.ingest")
        Invoke-Rag @("python", "scripts/clean_pages.py")
        Invoke-Rag @("python", "scripts/make_chunks.py")
        Invoke-Rag @("python", "-m", "rag.embed")
        Invoke-Rag @("python", "-m", "rag.index_faiss")
        Invoke-Rag @("python", "-m", "rag.index_bm25")
        Invoke-Rag @("python", "-m", "rag.search", $Query)
    }
}
