#!/bin/bash

echo "🔍 GitHub Pages 部署诊断"
echo "========================"
echo ""

echo "1️⃣ 检查 gh-pages 分支是否存在于远程..."
if git ls-remote --heads origin gh-pages &>/dev/null; then
    echo "   ✅ gh-pages 分支已推送到 GitHub"
    COMMIT=$(git ls-remote --heads origin gh-pages | awk '{print $1}')
    echo "   📝 最新提交: $COMMIT"
else
    echo "   ❌ gh-pages 分支不存在"
    exit 1
fi

echo ""
echo "2️⃣ 检查网站是否可访问..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" https://e1pco.github.io/lerobot_rdt/)
if [ "$HTTP_CODE" = "200" ]; then
    echo "   ✅ 网站返回 200 OK"
else
    echo "   ⚠️  网站返回 HTTP $HTTP_CODE"
fi

echo ""
echo "3️⃣ 检查本地构建时间..."
LOCAL_TIME=$(stat -c %Y _book/index.html 2>/dev/null || echo "0")
if [ "$LOCAL_TIME" != "0" ]; then
    echo "   📅 本地构建时间: $(date -d @$LOCAL_TIME '+%Y-%m-%d %H:%M:%S')"
else
    echo "   ⚠️  本地 _book 目录不存在"
fi

echo ""
echo "4️⃣ 检查远程部署时间..."
REMOTE_TIME=$(curl -s -I https://e1pco.github.io/lerobot_rdt/ | grep -i "last-modified" | cut -d' ' -f2-)
if [ ! -z "$REMOTE_TIME" ]; then
    echo "   📅 远程部署时间: $REMOTE_TIME"
else
    echo "   ⚠️  无法获取远程时间"
fi

echo ""
echo "5️⃣ 对比本地和远程内容..."
LOCAL_TITLE=$(grep -o '<title>[^<]*</title>' _book/ref_ik_robot_solver.html 2>/dev/null | head -n 1)
REMOTE_TITLE=$(curl -s https://e1pco.github.io/lerobot_rdt/ref_ik_robot_solver.html | grep -o '<title>[^<]*</title>' | head -n 1)
echo "   本地标题: $LOCAL_TITLE"
echo "   远程标题: $REMOTE_TITLE"

if [ "$LOCAL_TITLE" = "$REMOTE_TITLE" ]; then
    echo "   ✅ 标题匹配"
else
    echo "   ⚠️  标题不匹配（可能是缓存或部署延迟）"
fi

echo ""
echo "========================"
echo "📋 下一步建议："
echo ""
echo "如果网站无法访问或内容不对，请："
echo "1. 访问 https://github.com/E1pco/lerobot_rdt/settings/pages"
echo "2. 确认 'Source' 设置为 'Deploy from a branch'"
echo "3. 确认 'Branch' 选择了 'gh-pages' 和 '/ (root)'"
echo "4. 点击 'Save' 按钮"
echo "5. 查看部署状态: https://github.com/E1pco/lerobot_rdt/actions"
echo ""
echo "如果已经配置正确，等待 2-5 分钟后强制刷新浏览器 (Ctrl+Shift+R)"
