from pathlib import Path

import jinja2

from genai_evaluator.interfaces.interfaces import (
    LanguageType,
)


class TemplateStore:
    def __init__(
        self,
        dir_path: str,
        languages: tuple[LanguageType] = (
            LanguageType.FR,
            LanguageType.NL,
            LanguageType.EN,
        ),
        default_language: LanguageType = LanguageType.EN,
        do_validation: bool = False,
    ):
        """INITIALIZES THE GLOBAL TEMPLATE PRE-COMPUTER

        The TemplateStore loads all available templates into memory and sorts them by LanguageType.
        Sorting is determined based on file-path so watch out to avoid multiple directories named
        templates/
            english/
                example.jinja2
            french/
                example.jinja2
            dutch/
                example.jinja2

        THIS IS NOT:
            templates/
                english/
                    example.jinja2
                french/
                    example.jinja2
                dutch/
                    example.jinja2

        THIS store centralizes everything related to jinja2 templates and adds validation options that
        should only be activated during testing!

        Args:
            dir_path: path to the template-directory
            languages: tuple of languages to search for
            default_language: language to use by default
            do_validation: whether to validate the input parameters of all jinja templates or not during rendering
        """
        assert default_language in languages

        dir_path = Path(dir_path)

        self.jinja_env = (
            jinja2.Environment()
            if not do_validation
            else jinja2.Environment(undefined=jinja2.StrictUndefined)
        )
        self.default_language = default_language
        self.templates = {lang: {} for lang in languages}
        print(f"Loading templates from {dir_path}")
        for file_path in dir_path.glob("**/*.jinja2"):
            FILE_PATH_str = str(file_path).replace(str(dir_path), "")

            for lang in languages:
                if lang.value in FILE_PATH_str:
                    break
            else:
                raise ValueError(
                    f"Couldn't infer language of template file: {file_path}"
                )

            with file_path.open() as fp:
                txt = fp.read()

                if file_path.stem in self.templates[lang]:
                    raise ValueError(
                        f"A file named {file_path.stem} already exists, for the {lang} language"
                    )
                self.templates[lang][file_path.stem] = self.jinja_env.from_string(txt)

    def __getitem__(self, template_idx: tuple[LanguageType, str]) -> jinja2.Template:
        """RETURNS the appropriate jinja template based on filename, or language and filename

        Examples:
            template_store["my_template"] or template_store[LanguageType.EN, "my_template"]
            if the default is english

        Args:
            uses the default-language to index. If no language is given

        """
        if type(template_idx) is str:
            lang, template_name = self.default_language, template_idx
        else:
            lang, template_name = template_idx

        return self.templates[lang][template_name]
