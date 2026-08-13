# Sample retrieval test cases (Legixo sample corpus)

**Purpose:** Optional **gold-style** questions and answers for the **fictional**
markdown corpus (`01_` … `06_` files). Use these to check retrieval and
grounding against a known target. Machine-readable copy: `self_test.json`.

## In-corpus questions (answer from documents only)

| # | Question | Expected source file(s) |
|---|----------|-------------------------|
| 1 | What notice period applies when Bluecrest or Priya Nambiar ends the employment agreement? | `02_employment_agreement_excerpt.md` |
| 2 | How long is the non-compete after leaving Bluecrest, and when does it apply? | `02_employment_agreement_excerpt.md` |
| 3 | What kinds of information are called out as confidential in the Bluecrest excerpt? | `02_employment_agreement_excerpt.md` |
| 4 | What is the civil suit number and who are the parties in the transport invoice dispute memo? | `01_matter_memo_arvind_v_northfield.md` |
| 5 | Under the memo, what limitation period applies to contract claims under the fictional Riverside Code? | `01_matter_memo_arvind_v_northfield.md` |
| 6 | When is the next hearing in Arvind Mehta v. Northfield, and what is scheduled? | `01_matter_memo_arvind_v_northfield.md` |
| 7 | How many clear days before the listed date must parties file written arguments under the hearing notice rules? | `03_hearing_notice_template.md` |
| 8 | What time is case CV-2024-8812 listed, and what is it for? | `03_hearing_notice_template.md` |
| 9 | What happened to case CV-2023-4401 (Lakeview Society v. City Water Board), and what is the next date? | `03_hearing_notice_template.md` |
| 10 | For commercial suits above five lakh fictional rupees, what does Section 14 say about mediation? | `04_statute_style_excerpt_fictional.md` |
| 11 | If a contract fixes no interest rate, what rate may be awarded on admitted dues under Section 22? | `04_statute_style_excerpt_fictional.md` |
| 12 | What settlement offer did Northfield make in the counsel notes, and what counter-instruction did the client give? | `05_counsel_notes_settlement.md` |
| 13 | Are the settlement talks described in the counsel notes binding? What is the reminder? | `05_counsel_notes_settlement.md` |
| 14 | Who is the lessor and lessee for Unit 4B at Harbor View Tower, and what is the monthly rent? | `06_property_lease_clause.md` |
| 15 | What is the security deposit amount, and within how many days must it be refunded after handover? | `06_property_lease_clause.md` |
| 16 | Is subletting allowed for the Harbor View lease without extra steps? | `06_property_lease_clause.md` |

## Out-of-corpus questions (should not invent an answer from these docs)

| # | Question | Expected behavior | Why |
|---|----------|-------------------|-----|
| O1 | What is the population of Riverside city? | Say **not found in corpus** / **cannot answer** | Not in any sample file |
| O2 | What penalty applies if Priya breaches the non-compete? | **Not stated**; must not invent | Employment file has terms only, no penalty section |
| O3 | Who won case CV-2024-8812? | **Not stated**; dispute ongoing | Memo describes stage/hearing, not outcome |
| O4 | What is the LangGraph layout of this project? | **Not stated** | Corpus is legal notes only; no codebase content |

## How we use this in the repo

- Machine-readable cases live in `eval/self_test.json` (same data, for scripts).
- `eval/run_self_test.py` calls `POST /ask` for every case and records pass/fail
  in `eval/self_test.results.json`.
- Pass rule — in-corpus: `status=answered` **and** the expected source file
  appears in `citations`. Out-of-corpus: `status=not_found` **and** empty
  citations.