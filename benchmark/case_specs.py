"""配对实验的 CaseSpec:在 protocol 运行前一次性物化全部任务随机字段。

设计约束(2026-08-19 PAIRED BENCHMARK VALIDATION):
    同一 pair_id 的 V1/V3 两个 arm 必须读取同一份已持久化的 CaseSpec,
    不允许各自在 run 开始时重新抽随机数;initial state(如 S03 初始音量)
    同样由 spec 携带并在每个 arm 前显式 reset。本模块只服务 benchmark
    层,Agent production 不感知 CaseSpec 的存在。
"""

import json
import random
from dataclasses import asdict, dataclass, field
from typing import Any

from benchmark.tasks import (
    FIXTURE_CHAT_CONTACTS,
    FIXTURE_EMAIL_RECIPIENTS,
    FIXTURE_GALLERY,
    H01_SECTIONS,
    M01_TEMPLATES,
    QUERIES,
)

# M/H 任务的 CaseSpec 物化目录名占位:真实运行时 prepare 用实际
# desktop_dir 名重渲染 instruction;仅影响环境路径,不影响任务难度。
_CASE_DIR_NAME = "GUIAgentBenchmark_CASE"


@dataclass(frozen=True)
class CaseSpec:
    """保存一个完整任务实例的全部难度相关字段。

    Attributes:
        task_id: 任务编号(S01-S06)。
        pair_id: 配对标识(如 S03_P01);V1/V3 两臂共享同一 pair_id。
        case_seed: 生成本实例的种子(None 表示固定手工实例)。
        instruction: 发给 Agent 的逐字用户指令。
        params: 目标/期望/验证相关的全部随机字段(已物化)。
        initial_state: 每臂运行前需要 reset 的初始状态字段。
    """

    task_id: str
    pair_id: str
    case_seed: int | None
    instruction: str
    params: dict[str, Any] = field(default_factory=dict)
    initial_state: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """序列化为稳定 JSON 供报告保存与 V1/V3 一致性断言。"""
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


def _rng(task_id: str, pair_id: str, seed: int) -> random.Random:
    """按 (task_id, pair_id, seed) 派生独立确定性 RNG。"""
    return random.Random(f"{task_id}|{pair_id}|{seed}")


def _gen_s01(rng: random.Random) -> dict[str, Any]:
    """按 S01 原有四类运算模板物化表达式与期望结果。"""
    ops = [
        lambda: (rng.randint(10, 99), "+", rng.randint(10, 99)),
        lambda: (rng.randint(30, 99), "-", rng.randint(10, 29)),
        lambda: (rng.randint(3, 15), "×", rng.randint(3, 9)),
    ]
    if rng.random() < 0.25:
        divisor = rng.randint(2, 12)
        a, sym, b = divisor * rng.randint(4, 15), "÷", divisor
    else:
        a, sym, b = rng.choice(ops)()
    expression = f"{a}{sym}{b}"
    expected = eval(expression.replace("×", "*").replace("÷", "/"))
    return {
        "expression": expression,
        "expected": expected,
        "instruction": (
            f"打开系统计算器，计算{expression}，" "并让最终计算结果保留在计算器界面。"
        ),
    }


def _gen_s02(rng: random.Random) -> dict[str, Any]:
    names = ["晨星", "北辰", "青禾", "远山", "流光", "望舒"]
    words_en = ["Benchmark", "Agent", "Desktop", "GUI", "Testing", "Automation"]
    cn = rng.choice(names)
    en = rng.choice(words_en)
    num = rng.randint(100, 999)
    text = f"桌面 GUI 智能体测试 {cn}{en}{num}。"
    return {
        "text": text,
        "name_marker": cn,
        "number_marker": str(num),
        "instruction": (
            "打开系统文本编辑器(记事本)，新建一个空白文档，" f"并输入以下内容：{text}"
        ),
    }


def _gen_s03(rng: random.Random, initial: int, target: int) -> dict[str, Any]:
    return {
        "params": {"target_volume": target},
        "initial_state": {"volume": initial},
        "instruction": f"将系统输出音量调整到大约{target}%。",
    }


def _gen_s04(rng: random.Random) -> dict[str, Any]:
    apps = [
        ("Chrome浏览器", "chrome.exe", "Chrome_WidgetWin_1", None),
        ("文件管理器", "explorer.exe", "CabinetWClass", None),
        ("系统设置", None, None, "设置"),
    ]
    name, process, window_class, title_keyword = rng.choice(apps)
    return {
        "application": name,
        "process": process,
        "window_class": window_class,
        "title_keyword": title_keyword,
        "instruction": f"打开{name}，并让它的主窗口显示在桌面前台。",
    }


def _gen_s05(rng: random.Random) -> dict[str, Any]:
    query = rng.choice(QUERIES)
    return {
        "query": query,
        "instruction": f'使用当前浏览器搜索"{query}"，并停留在搜索结果页面。',
    }


def _gen_s06(_rng: random.Random) -> dict[str, Any]:
    return {
        "instruction": (
            "关闭桌面上那个测试专用的空白记事本窗口，" "不要关闭其他已经打开的窗口。"
        ),
    }


_GENERATORS = {
    "S01": lambda rng: _gen_s01(rng),
    "S02": lambda rng: _gen_s02(rng),
    "S03": lambda rng: _gen_s03(
        rng,
        initial=50,
        target=rng.choice([20, 35, 50, 65, 80]),
    ),
    "S04": lambda rng: _gen_s04(rng),
    "S05": lambda rng: _gen_s05(rng),
    "S06": lambda rng: _gen_s06(rng),
}


def generate_case(task_id: str, pair_id: str, seed: int) -> CaseSpec:
    """按 (task_id, pair_id, seed) 生成一次完整 CaseSpec;两臂共用结果。"""
    rng = _rng(task_id, pair_id, seed)
    fields = _GENERATORS[task_id](rng)
    instruction = fields.pop("instruction")
    params = fields.pop("params", fields)
    initial_state = fields.pop("initial_state", {})
    return CaseSpec(
        task_id=task_id,
        pair_id=pair_id,
        case_seed=seed,
        instruction=instruction,
        params=params,
        initial_state=initial_state,
    )


# S01/S03 paired validation 的固定实例:覆盖近距/下调/上调三档难度,
# 音量值全部位于 COM 标量接口可精确设置的 0-100 范围内。
def acceptance_cases() -> list[CaseSpec]:
    """PRD 全 15 项固定验收实例(2026-08-19 hardening;不随机、不重抽)。

    S01-S06 为 PRD 固定手工实例;M01-M05/H01-H03 由 generate_mh_case
    以固定种子物化全部随机字段;M06 为 SAFETY_SKIP 哨兵(破坏性系统
    操作不执行,报告单独列示)。
    """
    mh_ids = ["M01", "M02", "M03", "M04", "M05", "M06", "H01", "H02", "H03"]
    mh_cases = [
        generate_mh_case(task_id, f"PRD_{task_id}", seed=20260819) for task_id in mh_ids
    ]
    return [
        *_simple_cases(),
        *mh_cases,
    ]


def _simple_cases() -> list[CaseSpec]:
    """PRD 六个简单任务的固定手工实例。"""
    return [
        CaseSpec(
            task_id="S01",
            pair_id="PRD_S01",
            case_seed=None,
            instruction=("打开系统计算器，计算1+1，并让最终计算结果保留在计算器界面。"),
            params={"expression": "1+1", "expected": 2},
        ),
        CaseSpec(
            task_id="S02",
            pair_id="PRD_S02",
            case_seed=None,
            instruction=(
                "打开系统文本编辑器(记事本)，新建一个空白文档，"
                "并输入以下内容：Hello World"
            ),
            params={
                "text": "Hello World",
                "name_marker": "Hello",
                "number_marker": "World",
            },
        ),
        CaseSpec(
            task_id="S03",
            pair_id="PRD_S03",
            case_seed=None,
            instruction="将系统输出音量调整到大约60%。",
            params={"target_volume": 60},
            initial_state={"volume": 50},
        ),
        CaseSpec(
            task_id="S04",
            pair_id="PRD_S04",
            case_seed=None,
            instruction="打开Chrome浏览器，并让它的主窗口显示在桌面前台。",
            params={
                "application": "Chrome浏览器",
                "process": "chrome.exe",
                "window_class": "Chrome_WidgetWin_1",
                "title_keyword": None,
            },
        ),
        CaseSpec(
            task_id="S05",
            pair_id="PRD_S05",
            case_seed=None,
            instruction='使用当前浏览器搜索"Python"，并停留在搜索结果页面。',
            params={"query": "Python"},
        ),
        CaseSpec(
            task_id="S06",
            pair_id="PRD_S06",
            case_seed=None,
            instruction=(
                "关闭桌面上那个测试专用的空白记事本窗口，"
                "不要关闭其他已经打开的窗口。"
            ),
        ),
    ]


def _hex(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("0123456789ABCDEF") for _ in range(n))


def _gen_m01(rng: random.Random) -> dict[str, Any]:
    tpl = rng.choice(M01_TEMPLATES)
    headers = [str(h) for h in tpl["headers"]]
    rows = [[str(c) for c in r] for r in tpl["rows"]]
    header_text = "、".join(headers)
    rows_text = "；".join("、".join(r) for r in rows)
    return {
        "params": {"headers": headers, "rows": rows},
        "instruction": (
            "在表格软件(Excel)中新建一个工作簿，"
            f"第一行从A1单元格开始依次录入表头：{header_text}；"
            f"之后每行录一条数据，依次是：{rows_text}。"
            "全部录完后保持工作簿打开。"
        ),
    }


def _gen_m02(rng: random.Random) -> dict[str, Any]:
    recipient = rng.choice(FIXTURE_EMAIL_RECIPIENTS)
    tag = _hex(rng, 4)
    subject = f"资料确认 [{tag}]"
    body = f"测试资料已经收到，本轮编号为{_hex(rng, 4)}。"
    return {
        "params": {"recipient": recipient, "subject": subject, "body": body},
        "instruction": (
            "在当前浏览器打开的测试邮箱页面中写一封新邮件并发送。"
            f"收件人：{recipient}；主题：{subject}；正文：{body}"
        ),
    }


def _gen_m03(rng: random.Random) -> dict[str, Any]:
    target = rng.choice(FIXTURE_GALLERY)
    return {
        "params": {
            "image_title": target["title"],
            "filename": target["filename"],
        },
        "instruction": (
            f"把当前网页中标题为“{target['title']}”的那张图片，"
            f"保存到桌面的“{_CASE_DIR_NAME}”文件夹中。"
        ),
        "dir_instruction": True,
    }


def _gen_m04(rng: random.Random) -> dict[str, Any]:
    leader = rng.choice(["李明", "王芳", "张伟", "刘洋", "陈静"])
    numbers = [f"PX-{rng.randint(1000, 9999)}" for _ in range(3)]
    project_num = rng.choice(numbers)
    filename = f"项目说明_{_hex(rng, 3)}.txt"
    return {
        "params": {
            "leader": leader,
            "project_num": project_num,
            "numbers": numbers,
            "filename": filename,
        },
        "instruction": (
            f'打开桌面"{_CASE_DIR_NAME}"文件夹中的"{filename}"，'
            "找到其中的项目编号，并告诉我结果。"
        ),
        "dir_instruction": True,
    }


def _gen_m05(rng: random.Random) -> dict[str, Any]:
    contact = rng.choice(FIXTURE_CHAT_CONTACTS)
    message = f"GUI 测试编号 {_hex(rng, 4)} 已完成。"
    return {
        "params": {"contact": contact, "message": message},
        "instruction": (
            "在当前浏览器打开的聊天页面中，"
            f'找到联系人"{contact}"，发送消息：{message}'
        ),
    }


def _gen_h01(rng: random.Random) -> dict[str, Any]:
    sections = rng.sample(H01_SECTIONS, 3)
    target = rng.choice(sections)
    doc_title = f"文档_{_hex(rng, 3)}"
    return {
        "params": {
            "section_titles": [s["title"] for s in sections],
            "target_title": target["title"],
            "target_content": target["content"],
            "doc_title": doc_title,
        },
        "instruction": (
            f"把当前网页中标题为“{target['title']}”的那段正文，"
            "复制并粘贴到一个新建的 Microsoft Word 文档中，"
            f"并在第一行输入标题：{doc_title}"
        ),
    }


def _gen_h02(rng: random.Random) -> dict[str, Any]:
    title = f"桌面GUI智能体测试{_hex(rng, 3)}"
    tag = _hex(rng, 4)
    body = f"本轮测试编号为{tag}，用于验证视觉定位和文本输入能力。"
    return {
        "params": {"title": title, "body": body, "run_tag": tag},
        "instruction": (
            "在演示文稿软件(PowerPoint)中新建一个演示文稿。"
            "第一页创建一张包含标题和正文的幻灯片。"
            f"标题：{title}  正文：{body}"
        ),
    }


def _gen_h03(rng: random.Random) -> dict[str, Any]:
    count = rng.randint(5, 10)
    files = []
    for _ in range(count):
        ext = rng.choice([".pdf", ".txt", ".csv", ".doc"])
        prefix = rng.choice(["report", "notes", "image", "data", "manual"])
        files.append(f"{prefix}_{_hex(rng, 2)}{ext}")
    target = rng.choice(files)
    return {
        "params": {
            "files": files,
            "target_file": target,
            "directory": f"FileSearch_{_hex(rng, 3)}",
        },
        "instruction": (
            f'使用文件资源管理器，在桌面"{_CASE_DIR_NAME}"文件夹中，'
            f"找到文件“{target}”并打开。"
        ),
        "dir_instruction": True,
    }


def _gen_m06(_rng: random.Random) -> dict[str, Any]:
    return {
        "params": {"safety_skip": True},
        "instruction": "清空系统回收站中的测试项目，并完成确认操作。",
    }


_MH_GENERATORS = {
    "M01": _gen_m01,
    "M02": _gen_m02,
    "M03": _gen_m03,
    "M04": _gen_m04,
    "M05": _gen_m05,
    "M06": _gen_m06,
    "H01": _gen_h01,
    "H02": _gen_h02,
    "H03": _gen_h03,
}


def generate_mh_case(task_id: str, case_id: str, seed: int) -> CaseSpec:
    """按固定种子一次性物化 M/H 任务实例;字段冻结,运行期禁止重随机。"""
    rng = _rng(task_id, case_id, seed)
    fields = _MH_GENERATORS[task_id](rng)
    instruction = fields.pop("instruction")
    fields.pop("dir_instruction", None)
    params = fields.pop("params")
    return CaseSpec(
        task_id=task_id,
        pair_id=case_id,
        case_seed=seed,
        instruction=instruction,
        params=params,
    )


def fixed_pairs(task_id: str) -> list[CaseSpec]:
    """返回 S01/S03 paired validation 使用的固定 CaseSpec 列表。"""
    if task_id == "S01":
        fixed: list[tuple[str, object, object]] = [
            ("S01_P01", "23+48", 71),
            ("S01_P02", "72÷8", 9),
            ("S01_P03", "6×7", 42),
        ]
        return [
            CaseSpec(
                task_id="S01",
                pair_id=pair_id,
                case_seed=None,
                instruction=(
                    f"打开系统计算器，计算{expr}，" "并让最终计算结果保留在计算器界面。"
                ),
                params={"expression": expr, "expected": expected},
            )
            for pair_id, expr, expected in fixed
        ]
    if task_id == "S03":
        fixed = [
            ("S03_P01", 50, 60),
            ("S03_P02", 50, 20),
            ("S03_P03", 50, 80),
        ]
        return [
            CaseSpec(
                task_id="S03",
                pair_id=pair_id,
                case_seed=None,
                instruction=f"将系统输出音量调整到大约{target}%。",
                params={"target_volume": target},
                initial_state={"volume": initial},
            )
            for pair_id, initial, target in fixed
        ]
    raise ValueError(f"{task_id} 暂无固定配对实例定义")
