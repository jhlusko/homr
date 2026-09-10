"""Do the parser changes alter the token files other corpora produce?

music_xml_parser is shared by every converter - lieder, grandstaff, primus, pdmx,
musetrainer. If attaching notation changed a single byte of the token text, all five
would need re-converting.
"""
from homr.transformer.structured_notation import (
    BeamLevelState, NoteNotation, StemDirection, TieState,
    empty_beam_levels, empty_slur_slots,
)
from homr.transformer.vocabulary import EncodedSymbol
from training.transformer.training_vocabulary import token_lines_to_str

notation = NoteNotation(
    beam_levels=(BeamLevelState.BEGIN,) + empty_beam_levels()[1:],
    stem=StemDirection.UP,
    slurs=empty_slur_slots(),
    tie=TieState.START,
)
bare = [EncodedSymbol("clef_G2"), EncodedSymbol("note_8", "C5"), EncodedSymbol("barline")]
rich = [EncodedSymbol("clef_G2"), EncodedSymbol("note_8", "C5", notation=notation), EncodedSymbol("barline")]

print("token text identical:", token_lines_to_str(bare) == token_lines_to_str(rich))
print("symbols compare equal:", bare == rich)
print("hashes equal:", hash(bare[1]) == hash(rich[1]))
print("sample:", token_lines_to_str(rich).replace("\n", " | ")[:90])
