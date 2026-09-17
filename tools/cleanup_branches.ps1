#!/usr/bin/env pwsh
# Clean merged / zombie local branches. Run from the MAIN checkout by the owner.
# Protected: main, feat/dome-preview, and any branch currently checked out in a worktree.
$ErrorActionPreference = 'Continue'
$protected = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach ($b in @('main', 'feat/dome-preview')) { [void]$protected.Add($b) }

# Protect branches attached to existing worktrees
git worktree list --porcelain | ForEach-Object {
  if ($_ -match '^branch refs/heads/(.+)$') { [void]$protected.Add($Matches[1]) }
}

Write-Host "Protected: $($protected -join ', ')"
Write-Host "BEFORE local=$((git branch | Measure-Object).Count)"

$ok = 0; $force = 0; $skip = 0
foreach ($line in (git branch --format '%(refname:short)')) {
  $b = $line.Trim()
  if (-not $b -or $protected.Contains($b)) { continue }

  git branch -d $b 2>$null | Out-Null
  if ($LASTEXITCODE -eq 0) { $ok++; continue }

  git merge-base --is-ancestor $b origin/main 2>$null
  if ($LASTEXITCODE -eq 0) {
    git branch -D $b 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { $force++ } else { $skip++ }
  } else {
    $skip++
  }
}
Write-Host "Deleted safe=$ok force-ancestor=$force skipped=$skip"
Write-Host "AFTER local=$((git branch | Measure-Object).Count)"
Write-Host "--- remaining ---"
git branch

Write-Host "`n=== remote merged (preview; delete with -Apply) ==="
$apply = $args -contains '-Apply'
foreach ($line in (git branch -r --merged origin/main --format '%(refname:short)')) {
  $b = $line.Trim()
  if ($b -in @('origin/main', 'origin/HEAD', 'origin/archive/platform-layer', 'origin/feat/dome-preview')) { continue }
  $name = $b -replace '^origin/', ''
  if ($apply) {
    git push origin --delete $name
    Write-Host "push-deleted $name"
  } else {
    Write-Host "would delete remote $name"
  }
}
