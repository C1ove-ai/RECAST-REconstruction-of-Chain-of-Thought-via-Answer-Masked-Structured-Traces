param(
    [ValidateSet("project-dsh", "project-claude", "project-agents", "codex", "claude", "dsh", "agents", "all")]
    [string[]]$Targets = @("project-dsh", "project-claude"),
    [string]$WorkspaceRoot = ""
)

$skillName = "recast-cot-reverse"
$skillSrc = Join-Path $PSScriptRoot $skillName
if (-not (Test-Path -LiteralPath $skillSrc)) {
    Write-Error "skill source not found: $skillSrc"
    exit 1
}

if (-not $WorkspaceRoot) {
    # skills/ -> repository root
    $WorkspaceRoot = Split-Path $PSScriptRoot -Parent
}

function Install-Skill([string]$Root, [string]$Label) {
    $dest = Join-Path $Root $skillName
    New-Item -ItemType Directory -Force -Path $Root | Out-Null
    if (Test-Path -LiteralPath $dest) {
        Remove-Item -LiteralPath $dest -Recurse -Force
    }
    Copy-Item -LiteralPath $skillSrc -Destination $dest -Recurse -Force
    Write-Host "[install] $Label -> $dest"
}

if ($Targets -contains "all") {
    $Targets = @("project-dsh", "project-claude", "project-agents", "codex", "claude", "dsh", "agents")
}

foreach ($target in $Targets) {
    switch ($target) {
        "project-dsh"    { Install-Skill (Join-Path $WorkspaceRoot ".dsh\skills") "project DSH" }
        "project-claude" { Install-Skill (Join-Path $WorkspaceRoot ".claude\skills") "project Claude Code" }
        "project-agents" { Install-Skill (Join-Path $WorkspaceRoot ".agents\skills") "project shared agents" }
        "codex"          { Install-Skill (Join-Path $env:USERPROFILE ".codex\skills") "Codex user" }
        "claude"         { Install-Skill (Join-Path $env:USERPROFILE ".claude\skills") "Claude Code user" }
        "dsh"            { Install-Skill (Join-Path $env:USERPROFILE ".dsh\skills") "DSH user" }
        "agents"         { Install-Skill (Join-Path $env:USERPROFILE ".agents\skills") "shared agents user" }
    }
}

Write-Host ""
Write-Host "Installed '$skillName'. Start a new agent session (or reload skills) and say:"
Write-Host '  "帮我用 math500 跑一下 claude-sonnet-5-5 的 CoT 逆向"'
