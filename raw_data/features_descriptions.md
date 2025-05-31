All features are gotten from the experiences of the founder before starting the company used to
classify him/her as successful or not.
`founder
_
name
`
:
Name of the founder
`founder
_
linkedin
_
url`
:
LinkedIn URL of the founder
`founder
_
cb
_
url`
:
Crunchbase URL of the founder
`founder
_
twitter
_
url`
:
Twitter URL of the founder
`
cleaned
_
founder
_
linkedin
_
data
`
:
Linkedin data of the founder. This excludes achievements acquired before founding the
company used to classify the founder as successful or not. It also excludes fields that we cannot
be sure they are as before he founded the company.
`
cleaned
_
founder
_
cb
_
data
`
:
Crunchbase data of the founder. This excludes achievements acquired before founding the
company used to classify the founder as successful or not. It also excludes fields that we cannot
be sure they are as before he founded the company.
`
org_
name
`
:
Name of the organization we used to classify the founder as successful or not.
`
org_
cb
_
uuid`
:
Crunchbase UUID of the organization we used to classify the founder as successful or not.
`
org_
started
_
on
`
:
Date the organization we used to classify the founder as successful or not was founded.
`
org_
state
`
:
State the organization we used to classify the founder as successful or not is located in.
`
org_
city
`
:
The city the organization we used to classify the founder as successful or not is located in.
`
org_
description
`
:
Description of the organization we used to classify the founder as successful or not.
`
org_
category_
list`
:
Categories of the organization we used to classify the founder as successful or not.
`
org_
category_groups
_
list`
:
Category groups of the organization we used to classify the founder as successful or not.
`
professional
_
athlete
`
:
Boolean value indicating if the founder was a professional athlete or not.
`
childhood
_
entrepreneurship
`
Boolean value indicating if the founder was a childhood entrepreneur or not.
`
competitions
`
:
Boolean value indicating if the founder participated in 2 or more competitions or not.
`ten
_
thousand
_
hours
_
of
_
mastery
`
:
Boolean value indicating if the founder spent 10,000 hours mastering a hard/competitive skill or
not.
`languages
`
:
String (list) indicating the languages the founder speaks. Hint: You can treat this as a categorical
variable. E,g Speaking Spanish might be more valuable than speaking English.
You could also translate this to a numerical value by counting the number of languages the
founder speaks. Empty strings could be interpreted as 1 language.
`
perseverance
`
:
The founder should have a clear track record of sticking with projects and roles through
challenges, with descriptions highlighting overcoming significant obstacles (e.g.,
'Led the
company through a tough financial period, ultimately securing a successful Series B round').
This is high perseverance. When the founder has frequently changed jobs, with many roles
lasting less than a year, and descriptions lack details on challenges or long-term achievements,
this is low perseverance.
Allowed values: 0, 1, and 2.
`
risk
_
tolerance
`
:
Willingness to take risks in their career or business ventures. The founder left a senior position
at a Fortune 500 company to start a fintech startup targeting an unproven market segment is a
good example of high risk tolerance. If the founder has spent their career at large, stable
companies, progressing through well-defined roles without any involvement in startups or
high-risk projects, that is low risk tolerance.
Allowed values: 0, 1, and 2.
`
vision
`
:
Highlighting the ability to see the bigger picture and work or lead on innovative projects and
technologies before starting a company is a signal of vision. If the founder managed several
projects in established industries, primarily focusing on maintaining and optimizing existing
systems without driving major innovations, that is a negative signal for vision. These profiles are
dependable and detail-oriented but do not mention any visionary qualities or forward-thinking
achievements.
Allowed values: 0, 1, and 2.
`
adaptability
`
:
Highlighting the ability to quickly adapt to new roles and environments, emphasizing the
flexibility and willingness to embrace change is a positive signal. For instance, transitioning from
a technical role in a traditional industry to a leadership position in a fast-paced tech startup is a
signal of adaptability. If the founder has spent their career in a single industry, holding similar
roles across different companies with little variation in responsibilities or challenges, that is
negative for adaptability.
Allowed values: 0, 1, and 2.
`
personal
_
branding
`
:
Public Speaking or Thought Leadership: Does the founder mention speaking engagements,
industry panels, or thought leadership activities? Published Work or Media Appearances: Are
there mentions of published articles, books, or media contributions? Leadership Roles with
Public Focus: Does the founder hold or mention roles involving external engagement, such as
being a spokesperson or leading PR efforts? Recognition and Awards: Are there awards or
recognitions specifically related to public visibility or influence? Initiatives that Increase Visibility:
Does the founder mention initiating or leading projects that enhance their visibility or public
profile?
Allowed values: 0, 1, and 2.
`
education
_
level`
:
Highest degree attained by the founder.
associate
_
degree
_
or
_
no
_
degree
_
or
_
less = 0
bachelor
_
degree = 1
master
_
degree = 2
doctorate
_
or
_professional
_
degree
_
or
_
higher = 3
`
education
_
institution
`
:
Best institution the founder attended.
unknown
_
or
_
university_
ranked
_
after
_
2000 = 0
university_
ranked
_
500
_
2000 = 1
university_
ranked
_
100
_
500 = 2
university_
ranked
_
20
_
100 = 3
university_
ranked
_
in
_
the
_
top_
20 = 4
`
education
_
field
_
of
_
study
`
:
Field of study of the founder.
other = 0
social
_
sciences = 1
business
_
economics
_
finance = 2
stem = 3
`
education
_
international
_
experience
`
:
Boolean value indicating if the founder has international education experience or not.
`
education
_
extracurricular
_
involvement`
:
Boolean value indicating if the founder was involved in extracurricular activities during their
education or not. E.g, clubs, sports, etc.
`
education
_
awards
_
and
_
honors
`
:
Boolean value indicating if the founder received awards or honors during their education or not.
`big_
leadership
`
:
Roles held at any fortune 500 companies
held
_
c
_
level
_
roles
_
at
_
fortune
_
500
_
companies = 3
held
_
vp_
roles
_
at
_
fortune
_
500
_
companies = 2
held
_
director
_
at
_
fortune
_
500
_
companies = 1
no
_
leadership_
roles
_
at
_
fortune
_
500
_
companies = 0
`
nasdaq_
leadership
`
:
Roles held at any public tech companies
held
_
c
_
level
_
roles
_
at
_public
_
tech
_
companies = 3
held
_
vp_
roles
_
at
_public
_
tech
_
companies = 2
held
_
director
_
roles
_
at
_public
_
tech
_
companies = 1
no
_
leadership_
roles
_
at
_public
_
tech
_
companies = 0
`
number
_
of
_
leadership_
roles
`
:
Number of leadership roles held.
no
_
leadership_
roles = 0
held
_
one
_
leadership_
role = 1
held
_
more
_
than
_
one
_
leadership_
role = 2
`being_
lead
_
of
_
nonprofits
`
:
Boolean indicating being a lead of a non-profit organization.
`
number
_
of
_
roles
`
:
Integer total number of full-time roles held in the industry. If a founder moved up and changed
roles in the same company, you can count them as well. Do NOT consider mere internships,
part-time roles, or teaching/research assistant roles. ONLY consider full-time roles in the
industry!
`
number
_
of
_
companies
`
:
Integer total number of companies that they worked full-time for. Be careful if they have listed
multiple positions at the same company. These should only count as one company. Also be
careful about people listing internships, teaching-assistant, research assistant, angel investor,
jobs at university and other part time jobs as experience these shouldn't count as additional
companies. Do NOT consider mere internships, part-time roles, or teaching/research assistant
roles. ONLY consider full-time roles in the industry!
`industry_
achievements
`
:
Integer number of significant/notable achievements or awards in the industry.
`big_
company_
experience
`
:
Boolean, if the founder has worked at a big company. Big company can be defined as any
company that is in the NASDAQ 100 Technology Sector (NDXT) when the founder worked at
that company.
`
nasdaq_
company_
experience
`
:
Boolean, if the founder has worked at a public tech company. Public tech company can be
defined as any company that is in the NASDAQ 100 Technology Sector (NDXT) when they work
at that company.
`big_
tech
_
experience
`
:
Boolean, if the founder has worked at a big tech company. Big Tech can be defined as any
company that is in the NASDAQ 100 Technology Sector (NDXT) when the founder worked at
that company.
`
google
_
experience
`
:
Boolean, if the founder has worked at Google.
`facebook
_
meta
_
experience
`
:
Boolean, if the founder has worked at Facebook or Meta.
`
microsoft
_
experience
`
:
Boolean, if the founder has worked at Microsoft.
`
amazon
_
experience
`
:
Boolean, if the founder has worked at Amazon.
`
apple
_
experience
`
:
Boolean, if the founder has worked at Apple.
`
career
_growth`
:
Boolean, if the founder has shown significant continuous growth in job profile Eg. bachelor's to
Phd, and then going to engineer to VP of engineering in top companies.
`
moving_
around`
:
Boolean, if the founder changes full-time jobs too frequently. Example is that he works 1 year in
company X. Then he works at another company Y for 1 year.
`international
_
work
_
experience
`
:
Boolean, if the founder has worked full-time in 2 or more different countries.
`
worked
_
at
_
military
`
:
Boolean, if the founder has worked at the military.
`big_
tech
_position
`
:
If they worked as an engineer, researcher, product manager or other roles at a BigTech
company. Big Tech can be defined as any **Technology** company that is in the NASDAQ 100
Technology Sector (NDXT) when the founder worked at that company. Make sure this position is
at BigTech company. If they have worked as one of these positions in a non BigTech company it
doesn't count and that should be marked as 0.
researcher = 5
engineer = 4
product
_
manager = 3
sales
_
marketing = 2
other = 1
non
_
bigtech = 0
`
worked
_
at
_
consultancy
`
:
If they worked at a consultancy.
worked
_
at
_
mckinsey_
bcg_
bain = 3
worked
_
at
_
mid
_
tier
_
consultancies = 2
worked
_
at
_
unknown
_
consultancies = 1
never
_
worked
_
in
_
consultancy = 0
`
worked
_
at
_
bank`
:
If they worked at a bank.
worked
_
at
_
top_
tier
_
bank = 3
worked
_
at
_
mid
_
tier
_
large
_
bank = 2
worked
_
at
_
unknown
_
medium
_
size
_
banks = 1
did
_
not
_
work
_
at
_
bank = 0
`
press
_
media
_
coverage
_
count`
:
The level of press or media coverage received by the individual.
no
_
significant
_press
_
or
_
media
_
coverage = 0
moderate
_press
_
or
_
media
_
coverage = 1
high
_press
_
or
_
media
_
coverage = 2
`
vc
_
experience
`
:
If the founder held a role at a VC firm.
founder
_
did
_
not
_
hold
_
a
_
role
_
at
_
a
_
vc
_
firm = 0
founder
_
had
_
a
_junior
_
role
_
in
_
a
_
vc
_
firm = 1
founder
_
had
_
a
_
senior
_
role
_
in
_
a
_
vc
_
firm = 2
`
angel
_
experience
`
:
If the founder made angel investments.
founder
_
made
_
no
_
angel
_
investments = 0
founder
_
made
_
between
_
1
_
to
_
10
_
angel
_
investments = 1
founder
_
made
_
more
_
than
_
10
_
investments = 2
`
quant
_
experience
`
:
If the founder was a quant investor.
founder
_
was
_
not
_
a
_quant
_
investor = 0
founder
_
was
_
a
_quant
_
at
_
an
_
investment
_
firm = 1
founder
_
was
_
a
_quant
_
at
_
a
_
reputable
_
investment
_
firm = 2
`board
_
advisor
_
roles
`
:
Boolean, if the founder held board or advisor roles at large companies or well-known startups.
Do not consider board or advisor roles at small or unknown companies.
`tier
_
1
_
vc
_
experience
`
:
Boolean, if the founder has worked at a tier 1 VC firm.
`
startup_
experience
`
:
Boolean, if the founder has worked at a startup.
`
ceo
_
experience
`
:
Boolean, if the founder has been a ceo before.
`investor
_quality_prior
_
startup
`
:
Assessment the investors of the founder's prior organizations if exist.
no
_prior
_
startup_
exists = 0
prior
_
startup_
exists
_
raised
_
money_
not
_
from
_
tier
_
1
_
vcs = 1
prior
_
startup_
raised
_
money_
from
_
tier
_
1
_
vcs = 2
`
previous
_
startup_
funding_
experience
`
:
Funding of previous startups if exist.
not
_
available = 0
funding_
less
_
than
_
3m = 1
funding_
3m
_
to
_
10m = 2
funding_
10m
_
to
_
50m = 3
more
_
than
_
50m
_
in
_
funding = 4
`
max
_
amount
_
raised`
:
Maximum amount raised by the founder's previous startups if exist.
`
experienced
_
funding_
rounds
`
:
String. Funding rounds of previous startups if exist. This comes as <funding_
round
_
type> -
<organization>
. E,g 'Seed Round - NeuReality'
.
You can parse this string and count the number of funding rounds or use it as a categorical
variable.
`
all
_job
_
ids
`
:
List of jobs of the founder. Each job is a dictionary with keys
`title
`
and `industry
`
. You can
explore if experience in certain industries with certain roles is more valuable than others.
You can also relate this to the industry or categories of the organization we used to classify the
founder as successful or not.
`
repeat
_
ideal
_
days
`
:
Integer. If the founder is a repeat founder, the number of days between the latest previous
organization and the current organization (organization we used to classify the founder as
successful or not).
`founder
_
of
_
nonprofit`
:
Boolean. If the founder is a founder of a non-profit organization.
`
previous
_
orgs
_
categories
`
:
Categories of the previous organizations of the founder. You can explore if experience in certain
categories is more valuable than others.
`
previous
_
orgs
_
max
_
num
_
founders
`
:
Integer. Maximum number of founders per organization of the founder's previous organizations.
`
org_
num
_
founders
`
:
The number of founders of the organization we used to classify the founder as successful or
not.
Boolean. If the founder has experience with IPOs. If he has sent a company to IPO.
`
max
_
ipo
_
amount
_
raised`
:
Maximum amount raised by the founder's previous startups if they exist. A value of -1 means
there are ipo(s) but the amount raised is unavailable.
`
num
_
acquisitions
`
:
`ipo
_
experience
`
:
Integer. Number of acquisitions of the founder's previous startups if they exist. E.g, 2 means the
founder has founded 2 or more organizations and 2 were acquired.
`
acquirers
`
:
List of acquirers of all the founder's previous startups if they exist.
`
max
_
acquisition
_
amount`
:
Maximum amount of acquisition of the founder's previous startups if they exist. You can bin this
into categories or use it as a continuous variable. A value or -1 means there are acquisition(s)
but the price hasn’t been disclosed.
`l0l3
_persona
`
:
The persona of the founder. This is the persona we classified the founder as in the previous
iteration.
L1 = "Assign this if the founder started their first business during university studies or within one
year after graduation.
"
L2
_
1 = "Started their first business within ten years of graduation and previously worked at an
unknown small or mid-sized company (less than 500 employees).
"
L2
_
2 = "Started their first business within ten years of graduation and previously worked at a
non-tech publicly listed large company (over 500 employees).
"
L2
_
3 = "Started their first business within ten years of graduation and previously worked at a
top-tier investment bank: Goldman Sachs, Morgan Stanley, JP Morgan. Bank of America isn't
part.
"
L2
_
4 = "Started their first business within ten years of graduation and previously worked at a
tech publicly listed large company. EXCLUDE: Alphabet/Google, Amazon, Apple,
Meta/Facebook, Microsoft.
"
L2
_
5 = "Started their first business within ten years of graduation and previously worked at
these companies: Alphabet/Google, Amazon, Apple, Meta/Facebook, Microsoft.
"
L2
_
6 = "Started their first business within ten years of graduation and previously worked at a
top-tier management consultancy. ONLY CONSIDER TOP-TIER LIKE: McKinsey, Boston
Consulting Group, Bain & Company. DO NOT INCLUDE: Accenture, Deloitte, or such class.
"
L3
_
1 = "Has a PhD from a top 20 university (QS World University Rankings 2023) in a STEM
subject.
"
L3
_
2 = "Worked in research departments as a researcher or scientist at ONLY these
companies: Alphabet/Google, Amazon, Apple, Meta/Facebook, Microsoft.
"
L3
_
3 = "Worked at a unicorn startup or publicly listed company when it was less than 5 years
old.
"
L3
_
4 = "Worked in engineering departments at Alphabet/Google, Amazon, Apple,
Meta/Facebook, Microsoft for more than 10 years: .
"
L3
_
5 = "Evaluate if the founder has over 10 years of product management experience at
Alphabet/Google, Amazon, Apple, Meta/Facebook, or Microsoft.
"
L3
_
6 = "Previously worked as a venture capitalist.
"
L3
_
7 = "Was ever an executive (Director, VP, SVP, or C-level) at ONLY any of these companies:
Alphabet/Google, Amazon, Apple, Meta/Facebook, Microsoft.
"
L0 = "Assign this if only the founder does not fit any of the above personas.
"
`
skill
_
relevance
`
Whether the founder’s skills (technical, functional, or operational) directly address the startup’s
core needs, irrespective of domain expertise.
3 - Strong relevance
2 - Moderate relevance
1 - Weak relevance
0 - No relevance
`
entrepreneural
_
dna
`
Whether the founder has prior startup experience in the same or a related domain.
Boolean. True if yes, False if no. Null if it’s a first time founder.
`domain
_
expertise
`
How well the founder's industry experience and education match the startup's domain.
3 - Strong alignment
2 - Moderate alignment
1 - Weak alignment
0 - No alignment:
`
same
_
school`
Boolean, True means the founders attended the same institution, False means they did not
attend the same institution, and Null means it’s a single founder.
`
same
_job`
Boolean, True means the founders worked in the same company before, False means they did
not work in the same company before, and Null means it’s a single founder.
`
same
_
startup
`
Boolean, True means the founders have co founded a company before, False means they have
not co founded a company before and Null means it’s a single founder.
`
years
_
together
`
Number. This indicates the number of years the founders have known one another. A value of -1
means they’ve known one another before but we do not know the exact number of years.
`
org_
total
_
funding_
usd`
:
The total funding of the organization we used to classify the founder as successful or not. This
can be the target variable.
`
success
`
:
Boolean. If the founder is successful or not. True if org_
total
_
funding_
usd > 500M, False
otherwise.