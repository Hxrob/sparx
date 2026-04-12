# Openclaw NYC Open Data Examples

Use these examples to shape response quality, escalation behavior, and accessibility-aware recommendations.

## Example 1: Public computer access nearby

**User input**
Need a place in Brooklyn where I can use a public computer today.

**Expected assistant behavior**
- Query Citywide Public Computer Centers (`sejx-2gn3`)
- Prioritize results by proximity and operating status fields (if available)
- Return 2-3 options with concise next actions

**Expected response shape**
```markdown
## What I found
- I found nearby public computer center options in Brooklyn from NYC Open Data.

## Best next step
- Start with [Center A], then call ahead to confirm same-day access.
- Backup: [Center B] if the first location is full.

## If this does not work
- Call 311 and ask for "public computer access locations in Brooklyn" and nearest alternatives.

## Sources checked
- sejx-2gn3 Citywide Public Computer Centers: directory of public computer access points.
```

## Example 2: SNAP center with reduced walking

**User input**
I need the easiest SNAP center to reach with my walker from the Bronx, with less walking.

**Expected assistant behavior**
- Query SNAP Centers (`tc6u-8rnp`) for candidate offices
- Apply accessibility ranking: less elevation change, fewer transfers, less walking
- If routing certainty is limited, ask one clarifying question and provide a provisional best option

**Expected response shape**
```markdown
## What I found
- I identified SNAP centers that are likely easier to reach from your area.

## Best next step
- Preferred option: [Center A] because it appears to require fewer transfers and shorter walking.
- Backup option: [Center B] if timing or access is better.

## If this does not work
- Contact 311 and say: "I need SNAP center guidance with mobility constraints and minimal walking."

## Sources checked
- tc6u-8rnp Directory of SNAP Centers: SNAP office locations and service info.
```

## Example 3: Bus stop shelter support for weather-sensitive travel

**User input**
I have a disability and need bus stops with shelters for a trip in Queens.

**Expected assistant behavior**
- Query Bus Stop Shelters (`qafz-7myz`)
- Combine with user trip context to surface sheltered stop options
- Avoid claiming full route accessibility unless data explicitly confirms it

**Expected response shape**
```markdown
## What I found
- I found bus stop shelter locations relevant to your Queens trip.

## Best next step
- Use the sheltered stop at [Stop A] first, then verify current service timing.
- Backup sheltered stop: [Stop B].

## If this does not work
- Ask 311 for "accessible transit alternatives with sheltered waiting points in Queens."

## Sources checked
- qafz-7myz Bus Stop Shelters (Map): shelter locations for weather-protected waiting.
```

## Example 4: No confident Open Data answer -> 311 escalation

**User input**
I need help with a complex home service issue and I am not sure which agency handles it.

**Expected assistant behavior**
- Attempt Open Data mapping across likely service categories
- If no confident match, clearly explain uncertainty
- Provide exact 311 request language and preparation checklist

**Expected response shape**
```markdown
## What I found
- I could not confirm a single agency from available Open Data with high confidence.

## Best next step
- Contact 311 and use this request framing: "[short issue description], location, urgency, accessibility needs."
- Keep your address/intersection, photos (if safe), and preferred callback method ready.

## If this does not work
- Ask for escalation to a supervisor or agency-specific referral.

## Sources checked
- Multiple NYC Open Data service directories with no high-confidence direct match.
```
