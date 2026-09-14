#!/usr/bin/env bash
# 执行者健康检查：看产出物，不看进程。
cd "$(dirname "$0")/.." || exit 1
echo "=== 开发线（远端分支 / PR）$(date '+%H:%M') ==="
git fetch -q --prune origin 2>/dev/null
for b in fix/issue-322-fisheye-source-check fix/issue-319-fake-timeout-test fix/issue-320-dry-run-no-io; do
  if git ls-remote --heads origin "$b" 2>/dev/null | grep -q .; then
    n=$(git rev-list --count "origin/main..origin/$b" 2>/dev/null)
    echo "  ✅ $b  已推送 ${n:-?} 提交"
  else
    echo "  ⏳ $b  尚未推送"
  fi
done
echo "  -- 开放 PR:"
gh pr list --state open --json number,title,mergeStateStatus -q '.[]|"     #\(.number) \(.mergeStateStatus) \(.title)"' 2>/dev/null || echo "     (查询失败)"
echo
echo "=== 调研线（产出文档）==="
R="D:/Github/_research"
for f in "$R/dome_venue/SPEC_EXTRACT.md" "$R/dome_pipeline/PROJECTION_WORKFLOW.md" \
         "$R/redraion/CLEANUP_MANIFEST.md" "$R/redraion/STORY_BREAKDOWN.md" \
         "$R/redraion/CONTENT_SPEC.md" "$R/local_video/FEASIBILITY.md"; do
  short=${f#"$R/"}
  if [ -f "$f" ]; then
    printf "  ✅ %-45s %s 行  %s\n" "$short" "$(grep -c '' "$f")" "$(date -r "$f" '+%H:%M')"
  else
    d=$(dirname "$f")
    if [ -d "$d" ]; then
      printf "  ⏳ %-45s 目录已建（%s 个文件），文档未出\n" "$short" "$(ls -1 "$d" 2>/dev/null|wc -l)"
    else
      printf "  ❌ %-45s 无任何产出\n" "$short"
    fi
  fi
done
