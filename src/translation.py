# from fairseq.data import BertDictionary

from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask


@register_task('translation_in_same_language')
class TranslationCodeBARTTask(TranslationTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        TranslationTask.add_args(parser)
        parser.add_argument('--langs', required=True, metavar='LANG',
                            help='comma-separated list of monolingual language, '
                                 'for example, "en,de,fr". These should match the '
                                 'langs from pretraining (and be in the same order). '
                                 'You should always add all pretraining language idx '
                                 'during finetuning.')
        parser.add_argument('--extra-lang-symbol', default='', type=str,
                            help='comma-separated list of monolingual language, '
                                 'for example, "en,de,fr". These should match the '
                                 'langs from pretraining (and be in the same order). '
                                 'You should always add all pretraining language idx '
                                 'during finetuning.')

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.langs = args.langs.split(",")
        self.extra_langs = args.extra_lang_symbol.split(",")
        for d in [self.src_dict, self.tgt_dict]:
            for l in self.langs:
                d.add_symbol("[{}]".format(l))
            for l in self.extra_langs:
                d.add_symbol("[{}]".format(l))
            d.add_symbol("<mask>")
