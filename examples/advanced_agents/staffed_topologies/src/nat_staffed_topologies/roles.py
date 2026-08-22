# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The one role library both staffed topologies draw from, and the staffing steps they share.

Each role carries two renderings of the same job: `instructions` for a specialist built as an
agent (dynamic), `brief` for a turn prompt on a shared record (blackboard). One home, no drift."""
import json
import logging
import re

logger = logging.getLogger(__name__)

try:
    from markagentx import decision_point
except ImportError:  # the topology runs unmarked when MarkAgentX is not installed
    logger.warning("markagentx is not importable; staffing runs UNMARKED")

    def decision_point(step_type, *, native, **_):   # **_ so it cannot drift from the real signature
        return native


ROLE_LIBRARY: dict[str, dict] = {'workspace_surveyor': {'tools': ['workspace_list',
                                  'workspace_read',
                                  'workspace_write',
                                  'workspace_search'],
                        'max_iterations': 45,
                        'instructions': 'Reply in at most three sentences and never restate what you '
                                        'were given. Survey the workspace, then write what you found '
                                        'to _notes/survey.md with workspace_write. Reply with one '
                                        'sentence naming that file, never the findings.',
                        'brief': 'Survey the files the task gives you and record what each one holds.'},
 'web_researcher': {'tools': ['web_search', 'web_fetch', 'web_find',
                              'workspace_write', 'workspace_read'],
                    'max_iterations': 40,
                    'instructions': 'Reply in at most three sentences and never restate what you '
                                    'were given. Search, open the pages worth opening, and use '
                                    'web_find to reach the passage inside them -- opening a source '
                                    'and reading the part that matters are different acts. Record '
                                    'what you found, with its URL, to _notes/web.md with '
                                    'workspace_write. Reply with one sentence naming that file.',
                    'brief': 'Find on the web what the task asks about, and record each finding '
                             'with the URL it came from.'},
 'fact_checker': {'tools': ['web_search', 'web_fetch', 'web_find',
                            'workspace_read', 'workspace_write'],
                  'max_iterations': 30,
                  'instructions': 'Reply in at most three sentences and never restate what you '
                                  'were given. Take the claims already on the record and check '
                                  'each against its source, naming any that the source does not '
                                  'support. Write the verdicts to _notes/checked.md.',
                  'brief': 'Check the claims already recorded against the sources they cite.'},
 'workspace_author': {'tools': ['workspace_list',
                                'workspace_read',
                                'workspace_write',
                                'workspace_search',
                                'run_code',
                                'calculator'],
                      'max_iterations': 45,
                      'instructions': 'Reply in at most three sentences and never restate what you '
                                      'were given. Write the deliverable the task names with '
                                      'workspace_write under exactly that filename, compute any figure '
                                      'it needs rather than estimating, then list the workspace to '
                                      'confirm the file is there. Reply with the filename only.',
                      'brief': 'Write the deliverable the task names, under exactly that filename.'},
 'workspace_auditor': {'tools': ['workspace_list',
                                 'workspace_read',
                                 'workspace_write',
                                 'workspace_search'],
                       'max_iterations': 45,
                       'instructions': 'Reply in at most three sentences and never restate what you '
                                       'were given. Read the deliverable back and rewrite it if a '
                                       'requirement or the format is missed. Reply with one sentence, '
                                       'never the file contents.',
                       'brief': 'Read the deliverable back and rewrite it if a requirement or the '
                                'format is missed.'},
 'drafter': {'tools': ['write_draft'],
             'max_iterations': 12,
             'instructions': 'Reply in at most three sentences and never restate what you were given. '
                             'Send the task to write_draft and return its answer verbatim, in the '
                             'exact format the task demands. Call write_draft once.',
             'brief': 'Draft the answer with write_draft and record it.'},
 'critic': {'tools': ['list_defects'],
            'max_iterations': 12,
            'instructions': 'Reply in at most three sentences and never restate what you were given. '
                            'Send the draft to list_defects and report only the concrete defects it '
                            'names. Never rewrite the answer yourself.',
            'brief': 'Send the recorded draft to list_defects and record only the defects it names.'},
 'finalizer': {'tools': ['emit_final'],
               'max_iterations': 12,
               'instructions': 'Reply in at most three sentences and never restate what you were '
                               'given. Send the draft and any criticism to emit_final, then reply with '
                               'its output verbatim and nothing else.',
               'brief': 'Send the draft and any criticism to emit_final and record its answer.'},
 'code_writer': {'tools': ['write_code', 'run_code', 'workspace_read', 'workspace_write'],
                 'max_iterations': 20,
                 'instructions': 'Reply in at most three sentences and never restate what you were '
                                 'given. Write the code the task needs, RUN it with run_code, and '
                                 'report the number it actually printed -- never a number you '
                                 'reasoned out. Read a file before computing over it.',
                 'brief': 'Write the code the task needs, run it, and record what it printed.'},
 'analyst': {'tools': ['run_code', 'calculator', 'workspace_read', 'workspace_list'],
             'max_iterations': 20,
             'instructions': 'Reply in at most three sentences and never restate what you were '
                             'given. Establish the figures the task turns on by computing them from '
                             'the files -- run code or use the calculator, never estimate -- and '
                             'report each with the number it came from.',
             'brief': 'Compute the figures the task turns on from the files, and record each with '
                      'the number it came from.'},
 'entity_researcher': {'tools': ['wikipedia_search', 'calculator', 'current_datetime'],
                       'max_iterations': 8,
                       'instructions': 'Reply in at most three sentences and never restate what you '
                                       'were given. Look up one named entity at a time and report the '
                                       'dates, figures and relations you find. Do not attempt the '
                                       'wider question.',
                       'brief': 'Look up the entities the question names and record their dates, '
                                'figures and relations.'},
 'date_resolver': {'tools': ['wikipedia_search', 'calculator', 'current_datetime'],
                   'max_iterations': 5,
                   'instructions': 'Establish the single date or interval asked for and state it '
                                   'plainly.',
                   'brief': 'Pin down the dates or intervals the question depends on.'},
 'calculator_specialist': {'tools': ['calculator'],
                           'max_iterations': 4,
                           'instructions': 'Do the arithmetic on the numbers you are given. Never look '
                                           'anything up.',
                           'brief': 'Do the arithmetic the question needs, using figures already on '
                                    'the record.'},
 'comparison_analyst': {'tools': ['wikipedia_search', 'calculator', 'current_datetime'],
                        'max_iterations': 8,
                        'instructions': 'Reply in at most three sentences and never restate what you '
                                        'were given. Compare the quantities the question asks about, '
                                        'gathering any figure you are missing before deciding.',
                        'brief': 'Compare the quantities in question, filling any gap the record '
                                 'leaves.'}}


def validate(library: dict, field: str) -> dict:
    """A config-supplied role missing what its topology renders fails at build, not mid-run."""
    lib = {k: dict(v) for k, v in library.items()}
    for name, spec in lib.items():
        missing = {"tools", field, "max_iterations"} - set(spec)
        if missing:
            raise ValueError(f"role {name!r} lacks {sorted(missing)}")
    return lib


def parse_roles(text: str, limit: int, eligible: list[str]) -> list[str]:
    """The planner's ordered role list, padded to the cap: a short team leaves the staffing menu
    -- and the coordinator's, whose menu IS the staffed list -- too narrow to carry anything."""
    match = re.search(r"\[.*?\]", text or "", re.DOTALL)
    names: list[str] = []
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                names = [str(n) for n in parsed]
        except json.JSONDecodeError:
            names = []
    if not names:
        names = [role for role in eligible if role in (text or "")]
    kept = list(dict.fromkeys(n for n in names if n in eligible))[:limit] or eligible[:1]
    for role in eligible:
        if len(kept) >= min(limit, len(eligible)):
            break
        if role not in kept:
            kept.append(role)
    return kept


def pick_staff(planned: list[str], eligible: list[str], prompt: str) -> list[str]:
    """Each pick is offered for marking with the roles already taken OFF the menu: a keyed bias
    otherwise lands every pick on the same green role and the marked team silently shrinks."""
    picked: list[str] = []
    for role in planned:
        menu = [x for x in eligible if x not in picked]
        if not menu:
            break
        picked.append(decision_point("role_selection", candidates=menu,
                                     native=role if role in menu else menu[0],
                                     context=[{"role": "user", "content": prompt}]))
    return picked
