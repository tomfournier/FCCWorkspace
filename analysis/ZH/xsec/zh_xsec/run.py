
from argparse import ArgumentParser, Namespace, BooleanOptionalAction
from typing import Mapping, Sequence

from .parsing import create_parser
from .logger import get_logger

LOGGER = get_logger('__name__')



# ================ #
# PARSER UTILITIES #
# ================ #

def create_parser_from_parsers(
    parsers: ArgumentParser | Sequence[ArgumentParser],
    group_names: str | Sequence[str],
    description: str | None = None
) -> ArgumentParser:
    '''Create a parser by copying arguments from one or more parsers.

    Each source parser is copied into its own argument group. When multiple
    parsers are provided, ``group_names`` must contain one name per parser.
    '''
    if isinstance(parsers, ArgumentParser):
        source_parsers = [parsers]
    else:
        source_parsers = list(parsers)

    if isinstance(group_names, str):
        source_group_names = [group_names]
    else:
        source_group_names = list(group_names)

    if len(source_parsers) != len(source_group_names):
        raise ValueError('parsers and group_names must have the same length')

    action_names = {
        '_StoreTrueAction':  'store_true',
        '_StoreFalseAction': 'store_false',
        '_StoreConstAction': 'store_const',
        '_AppendAction':     'append',
        '_CountAction':      'count',
        '_SubParsersAction': 'parsers',
    }
    parser = ArgumentParser(description=description)

    for source_parser, group_name in zip(source_parsers, source_group_names):
        group = parser.add_argument_group(group_name)
        for action in source_parser._actions:
            if not action.option_strings or action.dest == 'help':
                continue

            kwargs = {
                'default':  action.default,
                'help':     action.help,
                'type':     action.type,
                'choices':  action.choices,
                'nargs':    action.nargs,
                'const':    action.const,
                'required': action.required,
                'metavar':  action.metavar,
            }
            kwargs = {key: value for key, value in kwargs.items()
                      if value is not None}

            action_name = action_names.get(type(action).__name__)
            if isinstance(action, BooleanOptionalAction):
                kwargs['action'] = BooleanOptionalAction
            elif action_name is not None:
                kwargs['action'] = action_name

            group.add_argument(*action.option_strings, **kwargs)

    return parser


def get_argument_metadata(parser: ArgumentParser) -> dict[str, dict]:
    '''Return metadata for each option registered on a parser.

    The returned dictionary is useful for wrappers and documentation without
    changing the parser API used by existing scripts.
    '''
    metadata = {}
    for action in parser._actions:
        if not action.option_strings:
            continue
        metadata[action.dest] = {
            'flags':    list(action.option_strings),
            'choices':  list(action.choices) if action.choices is not None else None,
            'type':     action.type,
            'help':     action.help,
            'default':  action.default,
            'nargs':    action.nargs,
            'const':    action.const,
            'required': action.required,
            'metavar':  action.metavar,
            'action':   type(action).__name__,
        }
    return metadata


def get_extra_args(args: Namespace, parser: ArgumentParser | Mapping[str, object]) -> list[str]:
    '''Build command-line arguments accepted by ``parser`` from ``args``.

    Regular options are forwarded with their current value. ``store_true``
    options are forwarded only when true. Boolean optional options are
    forwarded only when their value differs from the parser default, using
    either ``--name`` or ``--no-name`` as appropriate.
    '''
    if isinstance(parser, Mapping):
        parser = create_parser(**parser)

    extra_args = []
    metadata = get_argument_metadata(parser)

    for name, details in metadata.items():
        if not hasattr(args, name):
            continue

        value  = getattr(args, name)
        action: str = details['action']
        flags: list[str] = details['flags']
        flag   = next((item for item in flags if item.startswith('--')), flags[0])

        if action == 'BooleanOptionalAction':
            if value == details['default']:
                continue
            if value:
                extra_args.append(flag)
            else:
                no_flag = next((item for item in flags if item.startswith('--no-')),
                               f'--no-{flag.removeprefix("--")}')
                extra_args.append(no_flag)
        elif action == '_StoreTrueAction':
            if value:
                extra_args.append(flag)
        elif action == '_StoreFalseAction':
            if not value:
                extra_args.append(flag)
        elif action == '_CountAction':
            extra_args.extend([flag] * (value or 0))
        elif isinstance(value, (list, tuple)):
            extra_args.append(flag)
            extra_args.extend(str(item) for item in value)
        elif value is not None:
            extra_args.extend([flag, str(value)])

    return extra_args



# =================== #
# NAMESPACE UTILITIES #
# =================== #

def update_namespace(args: Namespace, **updates: object) -> Namespace:
    '''Return a copy of ``args`` with arbitrary attributes updated.'''
    values = vars(args).copy()
    values.update(updates)
    return Namespace(**values)



# ===================== #
# LOG MESSAGE UTILITIES #
# ===================== #

def log_msg(status: str, script: str, **variables: object) -> None:
    '''Log a formatted stage status message.'''
    details = ' | '.join(f'{name} = {value}' for name, value in variables.items())
    msg = f'{status}: [{script}] {details}'
    length = len(msg) + 2
    LOGGER.info('=' * length + '\n' + msg.center(length) + '\n' + '=' * length)
