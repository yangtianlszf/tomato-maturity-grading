# GitHub 发布检查清单

- [ ] 确认仓库中不包含发票、个人信息、答辩源文件、平台私有材料。
- [ ] 确认 `.pt`、`.pth`、`.zip`、数据集图片等大文件未被提交。
- [ ] 将公开权重上传到 GitHub Releases 或其他网盘，并在 README 中补充链接。
- [ ] 将 `configs/sdses_ssp.example.yaml` 复制为本地配置后再运行训练。
- [ ] 检查 README 中奖项、职责和结果描述是否准确。
- [ ] 初始化 Git 仓库并提交：

```bash
git init
git add .
git commit -m "Initial public release"
git branch -M main
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```
