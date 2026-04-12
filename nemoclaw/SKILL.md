---
name: openclaw-nyc-open-data
description: Handle NYC service-information requests by querying NYC Open Data first, then routing to a precise 311 path when direct answers are missing or weak. Use when users ask for NYC social services, eligibility, locations, agency contacts, or next-step civic guidance.
---

# Openclaw NYC Open Data Requests

Use this skill to answer NYC civic and social-service questions with a privacy-first, mobile-friendly response style.

## Scope

- Focus on NYC Open Data and agency-published datasets as the primary evidence source.
- Provide clear, actionable answers for residents navigating services and city systems.
- Ignore voice-input handling for now; this skill is text-first.
- Prioritize accessibility-aware guidance for people with mobility or disability needs.

## Priority Datasets

Use these datasets as preferred starting points when relevant:

- `sejx-2gn3` Citywide Public Computer Centers (digital access, internet-enabled public computers)
- `tc6u-8rnp` Directory of SNAP Centers (food benefit support locations)
- `qafz-7myz` Bus Stop Shelters (transit comfort and weather-protected access points)

## Operating Principles

- Prioritize trusted local/government sources before secondary references.
- Do not invent eligibility rules, hours, addresses, or phone numbers.
- If confidence is low or evidence is incomplete, say so and escalate to a concrete 311 pathway.
- Keep responses concise, plain-language, and mobile-friendly.
- Respect user privacy: avoid collecting unnecessary personal data.

## Workflow

Copy this checklist and work through it:

```text
NYC Request Flow
- [ ] 1) Understand the user problem and constraints
- [ ] 2) Normalize intent into searchable civic terms
- [ ] 3) Query NYC Open Data and gather evidence
- [ ] 4) Rank and validate candidate answers
- [ ] 5) Return direct answer OR route to 311 resolution
- [ ] 6) Format for mobile readability
```

### 1) Understand the request

Extract:
- Core problem (housing, food, benefits, permits, safety, sanitation, etc.)
- Borough/neighborhood constraints when provided
- Urgency and timing signals (today, this week, emergency)

### 2) Normalize intent

Rewrite the user request into:
- Service category
- Likely city agency/program
- Search synonyms (for multilingual phrasing or informal wording)
- Accessibility constraints (mobility devices, walking tolerance, stairs, transfer limits)

### 3) Query NYC Open Data first

Use available NYC Open Data tools/datasets to retrieve:
- Program/service details
- Service locations and contact channels
- Eligibility indicators and operational constraints (when explicitly available)
- Accessibility-relevant fields (step-free indicators, shelter presence, route burden proxies)

Capture evidence for each candidate answer:
- Dataset/source name
- Key fields used
- Short relevance note

### 4) Validate and rank

Prefer candidates with:
- Direct match to the user’s stated need
- Recent/complete records
- Specific next steps (where to go, who to call, what to submit)
- Accessibility fit for stated constraints

For accessibility-focused requests, rank options using this order:
1. Minimize elevation change and stair dependence
2. Minimize required transfers
3. Minimize total walking distance
4. Prefer sheltered or weather-resilient waiting points when transit is involved

If no strong match exists, move to 311 routing.

### 5) 311 fallback path

When Open Data cannot confidently answer:
- Identify the most likely 311 service/request type
- Provide the exact next action (contact method, request framing, info to prepare)
- Explain why this route is the best available escalation

### 6) Respond in mobile-friendly format

Use short sections and bullets with minimal jargon.

## Response Template

```markdown
## What I found
- [1-2 sentence answer]

## Best next step
- [Primary action]
- [Backup action, if relevant]

## If this does not work
- [311 route and how to phrase the request]

## Sources checked
- [Dataset/source 1]: [why it matters]
- [Dataset/source 2]: [why it matters]
```

## Safety and Quality Guardrails

- Never provide legal, medical, or immigration advice as facts; direct users to official channels.
- Distinguish observed evidence from recommendations.
- If the user appears in immediate danger, prioritize emergency guidance.
- Be explicit about uncertainty and missing data.
- Do not claim a route is fully accessible unless the source data explicitly supports it.
- Ask one concise clarifying question when accessibility needs are ambiguous and materially affect recommendations.

## Example Triggers

Apply this skill when prompts include patterns like:
- "Where can I get help with rent in NYC?"
- "How do I report illegal dumping near me?"
- "What city program can help with food access?"
- "I cannot find the right department; what should I do next?"
- "I need the most accessible way to get to a SNAP center."
- "Find options with less walking and fewer transfers."
