---
title: A0 onboarding
date: 2025-06-14 16:54:20
tags:
  - Environment
categories:
  - Assignment
comments: false
---

本作业旨在帮助你熟悉编程环境、提交流程以及基本的 PyTorch 编程。通过完成它，你将确保开发环境配置正确，理解如何提交未来的作业，并加强 PyTorch 编程技能。

# Environment Setup

## Option 1: Local Setup

- **Python**: 3.10 或更高版本  
- **Packages**: 推荐通过以下命令安装所有必要的依赖项：  
  ```bash
  pip install -r requirements.txt
  ```
- **Optional**: 建议使用 Nvidia GPU 并安装 CUDA 12.0 或更高版本 ，否则某些功能可能无法正常运行（我们会尽最大努力确保硬件差异不会影响你的评分）。

{% note info %}
**注意**：不同作业的 requirements.txt 可能略有差异
{% endnote %}

## Option 2: Docker Setup

强烈建议使用来自 [Nvidia PyTorch Release](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html) 的 Docker 镜像（例如 [23.10](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-10.html#rel-23-10) 或更新的版本）作为基础环境，以避免依赖冲突。

TODO: 完善


# Code and Debug

## Coding

所有完成 Tasks 所需的文件都位于 `src/` 目录下，该目录是**唯一**会被作为 Python 模块导入的目录。因此，你需要注意以下几点：
- `__init__.py` 文件对于 Python 模块来说是必不可少的，我们已经在 `src/` 中为你初始化好了所有必要的 `__init__.py` 文件，因此如果你出于个人目的需要修改它们，请务必小心。
- 如果你有其他需要在模块内部导入的文件（例如 `utils.py`），请确保它们也都放在 `src/` 目录下，并使用相对导入方式，例如：`from .utils import *，from .common.utils import ... ` 等。

### TODO: Task A0

your tasks in a0

## Debugging

以下内容用于帮助你调试和 debug：

### Naive Debug Mode

- 我们会在 `test_toy.py` 中提供一些带有明确答案的测试用例，这对你是可见的。
- 建议在提交前，先在自己的机器上确保代码正确运行，可以使用以下命令进行测试：
```bash
pytest test_toy.py
```
- 你可以根据自己的调试需求自由修改 `test_toy.py` 文件，我们不会使用它（以及下面提到的 `test_with_ref.py`）来为你的代码打分。

### Deep Debug Mode

{% note warning %}
**注意**: `test_toy.py` 和 `test_with_ref.py` 中的测试**仅用于调试目的**，它们可能**并不代表**我们在评分时使用的 `test_score.py` 中的实际测试用例。因此，请特别注意处理不同情况，尤其是一些 **edge cases**。
{% endnote %}

# Submission

- 你需要通过 `git commit` 和 `git push` 将作业提交到该私有仓库的 **`main` 分支**，包含作业要求的指定源文件，并确保在 **hard deadline** 之前完成提交，否则 **逾期作业将被自动拒收**。
- 尽量 **不要推送不必要的文件**，尤其是像图片这样的大文件到仓库中。
- 如果你因为一些特殊问题错过了截止时间，请直接联系老师（见 [Contact](#Contact)）。

{% note info %}
我们会尽可能频繁地对你的中间提交进行预测试，以用于最终提交的评分参考。 
每次预测试后，我们**只会提供 score feedback**（见 [Feedback](#Feedback) 部分），以便你在 **ddl** 之前改进代码，争取更高的分数。
{% endnote %}

# Scoring

每个作业将根据评分范围 0~100 分进行评定。我们会下载你的代码，并通过运行 `test_script.sh` 脚本来执行 `test_score.py` 文件（该文件对你来说是不可见的空文件），在我们的本地机器上导入 Tasks 中指定的文件并运行一些测试用例。
- 如果你在可选时间限制（optional time limit）内通过了所有测试，你将获得最高分 100 分。
- 如果你在可选时间限制内未通过任何测试，或者程序运行出现异常，你将获得最低分 0 分。
- 如果你在可选时间限制内只通过了部分测试，则你将获得介于 0~100 分 之间的分数，该分数是你通过的所有测试用例所对应分值的总和，具体得分标准见下表。

| Test Case | Score | Other Info |
| --- | --- | --- |
| Task0 - Case1 | 20 |  |
| Task0 - Case2 | 20 |  |
| Task0 - Case3 | 20 |  |
| Task0 - Case4 | 20 |  |
| Task0 - Case5 | 20 |  |
| Total | 100 |  |

# Feedback

在评分完成后，我们会将你的得分情况以一个表格的形式写入一个新文件 `score.md` 中，并通过一个新的 commit 推送到你的仓库中，分支名为 `score-feedback`（这是一个临时分支，仅用于让你查看每次评分后各个测试用例的得分情况，**请不要将其用于其他目的**）。

`score.md` 文件内容示例：

| Test Case | Score | Status | Error Message |
| --- | --- | --- | --- |
| Task0 - Case1 | 20 | ✅ |  |
| Task0 - Case2 | 20 | ✅ |  |
| Task0 - Case3 | 20 | ✅ |  |
| Task0 - Case4 | 20 | ✅ |  |
| Task0 - Case5 | 20 | ✅ |  |
| Total | 100 | 😊 |  |


status icons 的含义如下:
  - ✅: passed the case
  - ❌: failed the case due to wrong answers
  - 🕛: failed the case due to timeout if the time limit is set
  - ❓: failed the case due to some exceptions (the error message will be shown at the corresponding `Error Message` cell)
  - 😊: all passed
  - 🥺: failed at least one case

# Contact

记得关注老师的 Bilibili 账号，UID 为 390606417，观看[线上课程](https://space.bilibili.com/390606417/lists?sid=3771310)。