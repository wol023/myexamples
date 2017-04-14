find . -name "*.py" -exec sed -i '' "s:^import Image:from PIL import Image:g" {} \;
find . -name "*.py" | xargs egrep "^import Image"
