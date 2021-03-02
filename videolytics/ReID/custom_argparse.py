"""In order to reconstruct the option string from the parsed arguments we need to implement specific functions"""


import argparse
import typing
import shlex
import functools


def set_logger(parser: argparse.ArgumentParser):
    parser.add_argument('--log', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], type=str,
                        help='Threshold for logger level')


def canonical_label(action: argparse.Action) -> typing.Optional[str]:
    if not action.option_strings:
        return None
    for option_string in action.option_strings:
        if option_string.startswith('--'):
            return option_string
    return action.option_strings[0]


def argument_string(argument_parser: argparse.ArgumentParser, parameters: typing.Union[argparse.Namespace, dict]) -> str:
    getvalue = functools.partial(getattr, parameters) if isinstance(parameters, argparse.Namespace) else parameters.get
    string_builder = list()
    for action in argument_parser._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        value = getvalue(action.dest)
        if value != action.default:
            if isinstance(action, argparse._StoreConstAction):
                string_builder.append(canonical_label(action))
            elif isinstance(action, (argparse._StoreAction, argparse._ArgumentGroup)):
                this_canonical_label = canonical_label(action)
                if this_canonical_label is not None:
                    string_builder.append(this_canonical_label)
                if isinstance(value, list):
                    string_builder.extend(map(str, value))
                else:
                    string_builder.append(str(value))
    return ' '.join(map(shlex.quote, string_builder))
