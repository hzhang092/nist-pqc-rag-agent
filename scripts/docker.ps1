param(
    [ValidateSet(
        "help",
        "build",
        "build-allinone",
        "serve",
        "serve-allinone",
        "ingest",
        "clean",
        "chunk",
        "embed",
        "index-faiss",
        "index-bm25",
        "search",
        "ask",
        "ask-agent",
        "eval",
        "test",
        "pipeline"
    )]
    [string]$Task = "help",
    [string]$Query = "ML-KEM key generation"
)

$ErrorActionPreference = "Stop"

function Invoke-Compose([string[]]$ComposeArgs, [string[]]$PrefixArgs = @()) {
    $fullArgs = @($PrefixArgs + $ComposeArgs)
    Write-Host "docker compose $($fullArgs -join ' ')" -ForegroundColor Cyan
    & docker compose @PrefixArgs @ComposeArgs
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose failed with exit code $LASTEXITCODE"
    }
}

function Invoke-App([string]$Service, [string[]]$PyArgs, [switch]$AllInOne) {
    $prefixArgs = @()
    if ($AllInOne) {
        $prefixArgs = @("--profile", "allinone")
    }
    $composeArgs = @("run", "--rm", $Service) + $PyArgs
    Invoke-Compose $composeArgs $prefixArgs
}

function Show-Usage() {
    Write-Host "Usage:"
    Write-Host "  ./scripts/docker.ps1 -Task build"
    Write-Host "  ./scripts/docker.ps1 -Task build-allinone"
    Write-Host "  ./scripts/docker.ps1 -Task serve"
    Write-Host "  ./scripts/docker.ps1 -Task pipeline"
    Write-Host "  ./scripts/docker.ps1 -Task search -Query 'Algorithm 19 ML-KEM.KeyGen'"
    Write-Host "  ./scripts/docker.ps1 -Task ask-agent -Query 'Compare ML-KEM and ML-DSA'"
}

switch ($Task) {
    "help" {
        Show-Usage
    }
    "build" {
        Invoke-Compose @("build", "api")
    }
    "build-allinone" {
        Invoke-Compose @("build", "allinone") @("--profile", "allinone")
    }
    "serve" {
        Invoke-Compose @("up", "--build", "api")
    }
    "serve-allinone" {
        Invoke-Compose @("up", "--build", "allinone") @("--profile", "allinone")
    }
    "ingest" {
        Invoke-App "allinone" @("python", "-m", "rag.ingest") -AllInOne
    }
    "clean" {
        Invoke-App "allinone" @("python", "scripts/clean_pages.py") -AllInOne
    }
    "chunk" {
        Invoke-App "allinone" @("python", "scripts/make_chunks.py") -AllInOne
    }
    "embed" {
        Invoke-App "allinone" @("python", "-m", "rag.embed") -AllInOne
    }
    "index-faiss" {
        Invoke-App "allinone" @("python", "-m", "rag.index_faiss") -AllInOne
    }
    "index-bm25" {
        Invoke-App "allinone" @("python", "-m", "rag.index_bm25") -AllInOne
    }
    "search" {
        Invoke-App "api" @("python", "-m", "rag.search", $Query)
    }
    "ask" {
        Invoke-App "api" @("python", "-m", "rag.ask", $Query)
    }
    "ask-agent" {
        Invoke-App "api" @("python", "-m", "rag.agent.ask", $Query)
    }
    "eval" {
        Invoke-App "api" @("python", "-m", "eval.run")
    }
    "test" {
        Invoke-Compose @("config")
        Invoke-App "api" @(
            "python",
            "-c",
            "import json; import torch; from api.main import app; from rag.service import health_status; print(app.title); print(torch.__version__); print(json.dumps(health_status(), indent=2, sort_keys=True))"
        )
    }
    "pipeline" {
        Invoke-Compose @("build", "allinone") @("--profile", "allinone")
        Invoke-App "allinone" @("python", "-m", "rag.ingest") -AllInOne
        Invoke-App "allinone" @("python", "scripts/clean_pages.py") -AllInOne
        Invoke-App "allinone" @("python", "scripts/make_chunks.py") -AllInOne
        Invoke-App "allinone" @("python", "-m", "rag.embed") -AllInOne
        Invoke-App "allinone" @("python", "-m", "rag.index_faiss") -AllInOne
        Invoke-App "allinone" @("python", "-m", "rag.index_bm25") -AllInOne
        Invoke-App "api" @("python", "-m", "rag.search", $Query)
    }
}
