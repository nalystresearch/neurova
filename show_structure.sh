#!/bin/bash
echo "=== NEUROVA PROJECT STRUCTURE ==="
echo ""
echo "📦 Root Directory (PyPI Package Files)"
ls -1 *.md *.py *.toml *.in LICENSE 2>/dev/null | head -15
echo ""
echo "🐍 Source Package"
echo "neurova/"
ls -1d neurova/*/ 2>/dev/null | sed 's/^/  /' | head -20
echo ""
echo "🧪 Tests (NOT in PyPI)"
echo "tests/"
find tests -maxdepth 1 -type f -name "*.py" | wc -l | xargs echo "  " "test files"
ls -1d tests/*/ 2>/dev/null | sed 's/^/  /'
echo ""
echo "🔧 Scripts (NOT in PyPI)"  
echo "scripts/"
ls -1 scripts/*.sh scripts/*.py 2>/dev/null | wc -l | xargs echo "  " "scripts"
echo ""
echo "📚 Documentation (NOT in PyPI)"
echo "docs/"
ls -1d docs/*/ 2>/dev/null | sed 's/^/  /'
echo ""
echo "🛠️ Build (NOT in PyPI)"
echo "build/"
ls -1 build/lib/*.so 2>/dev/null | wc -l | xargs echo "  " "compiled binaries in build/lib/"
