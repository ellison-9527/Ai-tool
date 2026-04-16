# core/skill_manager.py
import os
import yaml
import shutil
import re
from pathlib import Path

SKILLS_DIR = Path(__file__).parent.parent / "skills"

# 允许被读取并直接喂给大模型的文件后缀
TEXT_EXTENSIONS = {
    ".md", ".txt", ".py", ".json", ".yaml", ".yml", ".html", ".css",
    ".js", ".ts", ".sh", ".bat", ".csv"
}


def get_safe_folder_name(name):
    return re.sub(r'[\\/*?:"<>|]', "", name).strip().replace(" ", "_")


def ensure_skills_dir():
    if not SKILLS_DIR.exists():
        SKILLS_DIR.mkdir(parents=True, exist_ok=True)


def safe_read(path: Path) -> str:
    """【超级容错】尝试用 UTF-8 读取，如果遇到 Windows 编码冲突，自动降级为 GBK 强读"""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="gbk", errors="ignore")


def get_all_skills():
    ensure_skills_dir()
    all_skills = {}

    for skill_folder in SKILLS_DIR.iterdir():
        if skill_folder.is_dir():
            skill_md = skill_folder / "SKILL.md"
            if skill_md.exists():
                try:
                    content = safe_read(skill_md)
                    match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)', content, re.DOTALL)
                    if match:
                        metadata = yaml.safe_load(match.group(1)) or {}
                        prompt = match.group(2).strip()
                    else:
                        metadata = {}
                        prompt = content.strip()

                    # 扫描该扩展包下所有附带的文本文件
                    attached_files = []
                    for file_path in skill_folder.rglob("*"):
                        if file_path.is_file() and file_path.name != "SKILL.md":
                            if file_path.suffix.lower() in TEXT_EXTENSIONS:
                                try:
                                    file_content = safe_read(file_path)
                                    rel_path = file_path.relative_to(skill_folder)
                                    attached_files.append({
                                        "path": str(rel_path),
                                        "content": file_content
                                    })
                                except Exception:
                                    pass

                    name = metadata.get("name", skill_folder.name)
                    all_skills[name] = {
                        "folder": skill_folder.name,
                        "description": metadata.get("description", ""),
                        "prompt": prompt,
                        "path": str(skill_md),
                        "files": attached_files
                    }
                except Exception as e:
                    print(f"⚠️ 解析技能 [{skill_folder.name}] 出错: {e}")
    return all_skills


def get_skill_choices():
    return list(get_all_skills().keys())


def load_skill_detail(skill_name):
    skills = get_all_skills()
    if skill_name in skills:
        return skills[skill_name]["description"], skills[skill_name]["prompt"]
    return "", ""


def save_skill(old_name, new_name, desc, prompt, script_files=None, ref_files=None):
    if not new_name.strip():
        return False, "⚠️ 技能名称不能为空"

    skills = get_all_skills()
    if old_name and old_name != new_name and old_name in skills:
        old_path = SKILLS_DIR / skills[old_name]["folder"]
        if old_path.exists():
            shutil.rmtree(old_path)

    folder_name = get_safe_folder_name(new_name)
    target_dir = SKILLS_DIR / folder_name
    target_dir.mkdir(parents=True, exist_ok=True)

    scripts_dir = target_dir / "scripts"
    refs_dir = target_dir / "references"
    scripts_dir.mkdir(exist_ok=True)
    refs_dir.mkdir(exist_ok=True)

    # 【容错修复】安全的判定数组真值
    if script_files is not None and isinstance(script_files, (list, tuple)):
        for fpath in script_files:
            shutil.copy(fpath, scripts_dir / Path(fpath).name)

    if ref_files is not None and isinstance(ref_files, (list, tuple)):
        for fpath in ref_files:
            shutil.copy(fpath, refs_dir / Path(fpath).name)

    skill_content = [
        "---",
        f"name: {new_name}",
        f"description: {desc}",
        "enable: true",
        "---",
        "",
        prompt
    ]
    (target_dir / "SKILL.md").write_text("\n".join(skill_content), encoding="utf-8")
    return True, f"✅ 技能 [{new_name}] (含关联文件) 已成功封装"


def delete_skill(skill_name):
    skills = get_all_skills()
    if skill_name in skills:
        target_path = SKILLS_DIR / skills[skill_name]["folder"]
        if target_path.exists():
            shutil.rmtree(target_path)
        return True, "🗑️ 扩展包已物理删除"
    return False, "⚠️ 未找到该技能"


def get_skill_prompts(selected_names):
    skills = get_all_skills()
    combined = []
    for name in selected_names:
        if name in skills:
            skill = skills[name]
            xml_parts = [f'<skill_content name="{name}">']
            xml_parts.append(f"# Skill: {name}\n")
            xml_parts.append(skill["prompt"])

            if skill["files"]:
                xml_parts.append("\n<skill_files>")
                for f in skill["files"]:
                    xml_parts.append(f'<file path="{f["path"]}" type="text">\n{f["content"]}\n</file>')
                xml_parts.append("</skill_files>")

            xml_parts.append('</skill_content>')
            combined.append("\n".join(xml_parts))
    return combined