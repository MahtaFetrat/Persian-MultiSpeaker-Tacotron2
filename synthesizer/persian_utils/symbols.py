
"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for Persian.
"""
# from . import cmudict

_pad        = "_"
_eos        = "~"
_characters = "ءابتثجحخدذرزسشصضطظعغفقلمنهويِپچژکگیآۀةأؤإئًَُّ!(),-.:;?  ̠،…؛؟‌٪#üABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_–@+/\u200c"

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
#_arpabet = ["@' + s for s in cmudict.valid_symbols]

# Export all symbols:
symbols = [_pad, _eos] + list(_characters) #+ _arpabet
