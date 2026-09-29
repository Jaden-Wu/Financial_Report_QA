# 财报数据来源与复现说明

## 数据来源

本项目使用10家A股汽车上市公司的2025年年度报告PDF。原实验文件由数据提供者通过iFinD下载。2026年9月29日补充验证了10份报告的公开HTTPS下载地址：9份来自巨潮资讯网，江淮汽车报告来自公司官网。逐份下载内容与原实验文件的大小及SHA256完全一致；链接和校验值记录于 `report_sources.json`。

| 公司简称 | 本地文件名 |
|---|---|
| 北汽蓝谷 | 北汽蓝谷25.pdf |
| 比亚迪 | 比亚迪25.pdf |
| 广汽集团 | 广汽25.pdf |
| 江淮汽车 | 江淮25.pdf |
| 赛力斯 | 赛力斯25.pdf |
| 上汽集团 | 上汽25.pdf |
| 宇通客车 | 宇通25.pdf |
| 长安汽车 | 长安25.pdf |
| 长城汽车 | 长城25.pdf |
| 中通客车 | 中通25.pdf |

报告年度为2025年，不应将2025年发布的2024年年度报告当作同一报告期。选择报告时应核对报告封面的年度和完整报告标题。

## 使用已有PDF复现

将上述10份PDF按表中名称放入项目根目录的 `reports` 文件夹。安装依赖后，依次执行：

```powershell
python -m pip install -r requirements.txt
python download_reports.py
python extract_reports.py
python build_chunks.py
python build_index.py
python -m streamlit run app.py
```

`download_reports.py` 默认只检查本地PDF，不会发起下载，也不会覆盖已有PDF。它检查PDF文件头、大小和SHA256是否与 `report_sources.json` 中本次实验使用的文件一致。

对于已在本地完成解析和建索引的项目，只需最后一条命令即可启动网页，不需要重复执行全部数据处理程序。

## 从零自动下载财报

在项目根目录执行以下命令，即可从已验证的公开地址下载全部缺失报告，无需登录iFinD。脚本只使用Python标准库，不需要额外安装下载依赖。

```powershell
python download_reports.py --download
```

默认保存到 `reports` 文件夹。已有文件只校验，不覆盖；下载内容须通过PDF文件头、大小及SHA256检查才保存。遇到网络超时可再次执行同一命令，已完成文件会跳过。若公共公告链接失效，脚本会明确报错，不会用不一致版本替代。

为不改动原实验数据，可在独立目录测试下载：

```powershell
python download_reports.py --download --output-dir reports_check
```

下载完毕后按上一节执行解析、切块、建索引和启动网页。报告版本一致，因此可复用本次实验的文件命名和出处页码。
