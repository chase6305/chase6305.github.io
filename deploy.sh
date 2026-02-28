#!/bin/bash

# Hugo自动部署脚本

echo "开始构建Hugo网站..."
hugo

# 检查构建是否成功
if [ $? -ne 0 ]; then
    echo "Hugo构建失败！"
    exit 1
fi

echo "进入public目录..."
cd public

# 如果是首次部署，初始化git
if [ ! -d ".git" ]; then
    echo "首次部署，初始化git仓库..."
    git init
    git remote add origin https://github.com/chase6305/chase6305.github.io.git
    git checkout -b gh-pages
fi

echo "添加所有文件..."
git add .

echo "请输入提交信息（直接回车使用默认信息）："
read commit_msg
if [ -z "$commit_msg" ]; then
    commit_msg="update site $(date '+%Y-%m-%d %H:%M:%S')"
fi

echo "提交更改..."
git commit -m "$commit_msg"

echo "推送到GitHub..."
git push origin gh-pages

echo "部署完成！"
echo "访问: https://chase6305.github.io/"
