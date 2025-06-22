import re
from pathlib import Path
from textwrap import dedent

from docutils.core import publish_doctree

PARENT_DIR = Path(__file__).parent
REPO_ROOT = PARENT_DIR.parents[1]
LANG_DIR = PARENT_DIR / "lang"
INDEX_MD = PARENT_DIR / "index.md"


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

        content = START_DOCS_REGEX.sub("", content)
        content = END_DOCS_REGEX.sub("", content)
        content = HIDE_IN_DOCS_REGEX.sub("", content)
        content = REPLACE_IN_DOCS_REGEX.sub(r"\g<1>", content)

        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text(content)

    lang_toc_paths = []
    for lang in list(doc_mapping.values())[1:]:
        api_reference = publish_doctree(lang.with_name("api-reference.rst").read_text())
        title = api_reference.get("title").replace("MVSR ", "")
        lang_toc = f"""
            ```{{toctree}}
            :hidden:

            {title} <api-reference>
            ```
        """
        lang.write_text(lang.read_text() + dedent(lang_toc))
        lang_toc_paths.append(str(lang.relative_to(LANG_DIR.parent).with_suffix("").as_posix()))

    lang_toc_paths.sort(key=lambda path: path != "lang/cpp/index")

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


START_DOCS_REGEX = re.compile(r"(?:.|\n)*<!--start-docs-->")
END_DOCS_REGEX = re.compile(r"<!--end-docs-->(?:.|\n)*")
HIDE_IN_DOCS_REGEX = re.compile(r"<!--hide-in-docs-->\n^.*$", flags=re.MULTILINE)
REPLACE_IN_DOCS_REGEX = re.compile(r"<!--replace-in-docs (.*) -->\n^.*$", flags=re.MULTILINE)
