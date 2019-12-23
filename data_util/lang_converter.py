lang_short = {"german": "de", "english": "en", "turkish": "tr", "finnish": "fi", "russian": "ru", "spanish": "es",
              'arabic': "ar", 'bulgarian': "bg", 'french': 'fr', 'hindi': 'hi', 'urdu': 'ur', 'vietnamese': 'vi',
              'chinese': 'zh'}


def convert_long_to_short(lang):
    return lang_short[lang]


def convert_short_to_long(lang):
    for l in lang_short:
        if lang_short[l] == lang:
            return l