import re

from sphinx.application import Sphinx as App
from sphinx.ext.autodoc import ClassDocumenter


class QualNameClassDocumenter(ClassDocumenter):
    objtype = "class-qualname"

    def add_directive_header(self, sig: str):
        name = getattr(self.object, "__qualname__", self.objpath[-1])
        self.add_line(f".. py:class:: {name}{sig}", "", 0)


def build_finished(app: App, exception: Exception | None):
    if exception:
        return

    for html_file in app.outdir.glob("**/*.html"):
        content = html_file.read_text()
        content = FIX_KERNEL_REFS_REGEX.sub(FIX_KERNEL_REFS_SUB, content)
        content = FIX_KERNEL_METHOD_REFS_REGEX.sub(FIX_KERNEL_METHOD_REFS_SUB, content)
        content = FORCE_EXTREF_CODE_TAGS_REGEX.sub(FORCE_EXTREF_CODE_TAGS_SUB, content)
        html_file.write_text(content)


FIX_KERNEL_REFS_REGEX = re.compile(r"(Kernel)(?:\.\1)?\.([^.]+)\.\2")
FIX_KERNEL_REFS_SUB = r"\g<1>.\g<2>"

FIX_KERNEL_METHOD_REFS_REGEX = re.compile(
    r"<code [^>]*>.*?[^.]\b(normalize|denormalize)\(\).*?</code>"
)
FIX_KERNEL_METHOD_REFS_SUB = (
    r'<a class="reference internal" href="#mvsr.Kernel.Raw.\g<1>" title="Kernel.Raw">\g<0></a>'
)

FORCE_EXTREF_CODE_TAGS_REGEX = re.compile(
    r'(<a class="reference external" [^>]*>)(?:<(?!code\b)[^/][^>]*>)*([^<>]+)(?:</[^>]+>)*(</a>)'
)
FORCE_EXTREF_CODE_TAGS_SUB = (
    r"\g<1>"
    r'<code class="xref py py-class docutils literal notranslate">'
    r'<span class="pre">\g<2></span>'
    r"</code>"
    r"\g<3>"
)


def setup(app: App):
    app.add_autodocumenter(QualNameClassDocumenter)
    app.connect("build-finished", build_finished)
