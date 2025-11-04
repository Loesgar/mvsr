import re
import tomllib
from pathlib import Path
from textwrap import dedent
from urllib.parse import urljoin

from docutils.core import publish_doctree

PARENT_DIR = Path(__file__).parent
REPO_ROOT = PARENT_DIR.parents[1]
LANG_DIR = PARENT_DIR / "lang"
INDEX_MD = PARENT_DIR / "index.md"

PYPROJECT_TOML = REPO_ROOT / "lang" / "python" / "pyproject.toml"
BASE_URL = urljoin(
    tomllib.loads(PYPROJECT_TOML.read_text())["project"]["urls"]["Repository"] + "/", "tree/main/"
)


def generate_from_readmes():
    INDEX_MD.unlink(missing_ok=True)

    doc_mapping = {
        REPO_ROOT / "README.md": INDEX_MD,
        **{
            lang_readme: LANG_DIR / lang_readme.parent.name / "index.md"
            for lang_readme in (REPO_ROOT / "lang").glob("*/README.md")
        },
    }
    for source_file, target_file in doc_mapping.items():
        content = source_file.read_text()

        content = HIDE_IN_DOCS_REGEX.sub("", content)
        content = SHOW_IN_DOCS_REGEX.sub(r"\g<1>", content)

        for match in reversed(list(FIND_CROSS_REFERENCES_REGEX.finditer(content))):
            reference = (source_file.parent / match.group(1)).resolve()
            if doc_file := doc_mapping.get(reference):
                fixed = str(doc_file.relative_to(target_file.parent, walk_up=True))
            else:
                fixed = urljoin(BASE_URL, str(reference.relative_to(REPO_ROOT)))
                if fixed.split(".")[-1].lower() in {"jpg", "png", "svg", "webp"}:
                    fixed = fixed.replace("github.com/", "raw.githubusercontent.com/")
                    fixed = fixed.replace("/tree/", "/")
            content = content[: match.start(1)] + fixed + content[match.end(1) :]

        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text(content)

    lang_toc_paths: list[str] = []
    for lang in list(doc_mapping.values())[1:]:
        api_reference = publish_doctree(
            lang.with_name("api-reference.rst").read_text(),
            settings_overrides={"report_level": 5},
        )
        title = api_reference.get("title").replace("MVSR ", "")
        lang_toc = f"""
            ```{{toctree}}
            :hidden:

            {title} <api-reference>
            ```
        """
        lang.write_text(lang.read_text() + dedent(lang_toc))
        lang_toc_paths.append(str(lang.relative_to(LANG_DIR.parent).with_suffix("").as_posix()))

    lang_toc_paths.sort(key=lambda path: (path != "lang/cpp/index", path))

    toc = f"""\
        ```{{toctree}}
        :hidden:

        Overview <self>
        ```

        ```{{toctree}}
        :caption: Supported Languages
        :hidden:

        {"\n".ljust(9).join(lang_toc_paths)}
        ```\n
    """

    INDEX_MD.write_text(dedent(toc) + INDEX_MD.read_text())


HIDE_IN_DOCS_REGEX = re.compile(r"<!--hide-in-docs-->\n^.*$", flags=re.MULTILINE)
SHOW_IN_DOCS_REGEX = re.compile(r"<!--show-in-docs (.*) -->")
FIND_CROSS_REFERENCES_REGEX = re.compile(r"\]\(([^):#][^):#]*)(?:#[^)]+)?\)")
