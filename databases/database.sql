SQLite format 3   @    �   x                                                           � .j�� 	 
�.�	�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 �q�5tablepos_tagspos_tags4CREATE TABLE pos_tags(
                    id INTEGER PRIMARY KEY,
                    text TEXT UNIQUE,
                    tag TEXT,
                    dep TEXT,
                    head TEXT
                )/C indexsqlite_autoindex_pos_tags_1pos_tags5�(�#tableentitiesentities2CREATE TABLE entities(
                id INTEGER PRIMARY KEY,
                entity TEXT UNIQUE,
                label TEXT
            )/C indexsqlite_autoindex_entities_1entities3�,++�tabletokenized_wordstokenized_wordsCREATE TABLE tokenized_words (
                token INTEGER PRIMARY KEY,
                word TEXT NOT NULL UNIQUE
            )=Q+ indexsqlite_autoindex_tokenized_words_1tokenized_words          �    0����������������������������r]O?6' ����������pfZLC3&�����������vl]J?/$�����������zrf\SE:,
�
�
�
�
�
�
�
�
�
�
p
a
R
C
4
$


	�	�	�	�	�	�	�	�	�	�	s	g	Y	N	D	'		����������yh\ND8%���������zpdTE:"���������|i\PC2%����������}k^OB6+����������zjXH@8.$���������rhYOE;0�����������ui]LC1           � %displacement� air� #composition	� assess	� change
�
 measure�	 important
� compare� lbm� mass� lean
� prolong� vs� condition� nogueira	�  keeler� stem	�~ simply�} protocol�| 'heterogeneity�{ topic�z give�y good�x fast�w influence�v wilk�u recently�t #hypothesize�s !previously�r critical
�q finding�p +notwithstanding�o %hypertrophic�n lead�m #combination
�l analyze
�k similar	�j result�i wide�h find�g al�f et�e !schoenfeld�d %metaanalysis�c !systematic	�b tempos�a slow�` #detrimental�_ indicate
�^ support	�] poorly	�\ remain�[ strategy
�Z benefit	�Y manner�X %
super
slow
�W %professional	�V common�U order	�T action
�S combine�R inclusion
�Q usually�P %conventional
�O discuss�N main
�M provide�L approach�K %specifically
�J develop
�I current�H endeavor�G date�F paper�E scholarly�D comprise�C read
�B optimal�A #extrapolate
�@ section	�? method
�> outline�= '
proper
form

�< examine�; study�: previous�9 similarly�8 ambiguity	�7 create	�6 define�5 !explicitly
�4 mention	�3 author�2 infer�1 stage�0 early�/ 1
proper
technique
�. learn�- highlight�, !researcher	�+ depend�* vary�) %prescription�( stand	�' scarce
�& outcome
�% explore�$ directly
�# prevent�" 'effectiveness�! emphasis�  practice� prescribe� alignment� 'orchestration	� injury� risk� minimize	� target� #effectively	� ensure	� bodily� execution
� control
� pertain� !standpoint� power� #potentially� goal� specific� !individual	� affect
� heavily�
 ;
appropriate
technique
�	 note� worth� position	� stance� -widthorientation� grip
� correct
� involve
� primary
�  include checklist	~ manual} nsca| #association{ %conditioningz strengthy nationalx #constituentw !literaturev !scientificu !definitiont !agreedupons #universallyr currently
q proposep differento describen 5
training
technique
m terml effectivek componentj keyi refer
h failureg muscularf momentarye proximity	d effortc intensityb weeka group
` perform_ set^ ie	] volume
\ certain[ %manipulation
Z requireY knowX %introduction	W growthV #developmentU form
T keywordS variation
R lenient	Q versus	P strictO #investigate	N futureM )recommendationL !accordanceK adopt	J followI universal
H suggest
G limited	F effect	E aspect	D impactC #biomechanic
B anatomyA apply	@ theory? imply> base= generally
< pattern; movement: #positioning9 body8 guideline
7 enhance6 phase5 !concentric4 eccentric3 duration2 !manipulate1 determine0 need/ research. s	- length, long	+ employ* recommend) rom	( motion' range& tempo% !repetition$ type# #contraction" kinematic! -exercisespecific  variable focus	 proper !constitute evidence exist !synthesize aim	 review narrative #hypertrophy eg !adaptation maximize try !especially technique exercise #appropriate !importance emphasize %practit   /�l   -�w   +�   )�   &�0   %�S   #�H    �N   �H   �D   �@   �t   �p   �k   �g   �Z   
�V   �   �
   �w 	y/   .�	���{�p_��(�F	y��P�^	�����D�j	��C �
�Mq�	�
%�@�l	����
���
Se
�����
��~2���
�i�;��o���[z&A�}�C^���3P/��w�@<�F	�]�M	F���{��`�	�P���Z9F�0	��
��P����;��
5�����	��v�YQO	'��
��	E�	�&&,�]�g�g�	Z4���}{���	����
��-s�K
���(��U��E���I�
DDs������	h����
%����k]5{
��1�7�
q_s q��T%	ti	O            %displacementair#compositionassesschangemeasure
important	comparelbm	mass	leanprolongvsconditionnogueirakeeler 	stem �simply �protocol �'heterogeneity �
topic �	give �	good �	fast �influence �	wilk �recently �#hypothesize �!previously �critical �finding �+notwithstanding �%hypertrophic �	lead �#combination �analyze �similar �result �	wide �	find �al �et �!schoenfeld �%metaanalysis �!systematic �tempos �	slow �#detrimental �indicate �support �poorly �remain �strategy �benefit �manner �%
super
slow
 �%professional �common �
order �action �combine �inclusion �usually �%conventional �discuss �	main �provide �approach �%specifically �develop �current �endeavor �	date �
paper �scholarly �comprise �	read �optimal �#extrapolate �section �method �outline �'
proper
form
 �examine �
study �previous �similarly �ambiguity �create �define �!explicitly �mention �author �
infer �
stage �
early �1
proper
technique
 �
learn �highlight �!researcher �depend �	vary �%prescription �
stand �scarce �outcome �explore �directly �prevent �'effectiveness �emphasis �practice �prescribe �alignment �'orchestration �injury �	risk �minimize �target �#effectively �ensure �bodily �execution �control �pertain �!standpoint �
power �#potentially �	goal �specific �!individual �affect �heavily �;
appropriate
technique
 �	note �
worth �position �stance �-    ,tributing   �r�e
type
outcome
eg
bodyfat
analysis
suggest
trivial
smd
ci
–
favour
figure
display
individual
effect
size
tick
posterior
probability
distribution
overall
estimate
outcome
study
design
study
categorize
subject
design
eg
subject
different
range
motion
different
limb
betweensubject
design
eg
subject
assign
perform
intervention
prom
intervention
withinparticipant
design
analysis
reveal
small
smd
ci
–
favour
outcome
participant
design
analysis
reveal
trivial
smd
ci
–
favour
figure
display
individual
effect
size
tick
posterior
probability
distribution
overall
estimate
outcome
figure
overall
model
figure
outcome
subgroup
analysis
figure
study
design
subgroup
analysis
proximal
vs
distal
muscle
hypertrophy
hypertrophy
outcome
assessment
group
c   �Z�5
ie
muscle
length
origin
regional
muscle
hypertro
phy
assessment
method
proximal
muscle
hypertrophy
trivial
smd
ci
–
find
favour
dis
tal
muscle
hypertrophy
small
smd
ci
–
find
favour
individual
effect
size
posterior
probability
distribution
overall
subgroup
estimate
find
figure
resistance
training
modality
resistance
training
intervention
categorize
resistance
machine
free
weight
combination
intervention
ex
clusively
resistance
machine
subgroup
analysis
reveal
trivial
smd
ci
�   �    construct�   
cause�   	#antioxidant�v:c��   "
lunge�   discrep`��pi  � 	grow��mmr�k�? 
O�     halflive�   '
scale�qG   1perforc8on#y��     (+stretchactivate�   !innate!   expend7� rotation	�lH� locate*�te�     $%proportional� traverse�   *!myopathiesp�   triconly	 
 �  � ����J� �i*��XX9; � �'  � ���wi\N>. � �m�[
muscle
length
average
assume
muscle
length
im'
firstinline

s	- A
increase
individual
muscle
size
contribute
increase
individ
classic
@

hub

l5
hypertrophy
sensor

H9
hypertrophy
stimulus

FA�
hypertro
phy
training
periodize
goal
maximize
muscle
mass
	ܖ\�9
condition
lateral
medial
head
triceps
brachii
see
great
hypertrophy
contrast
study
study
stasinaki
et
al
find
significant
difference
triceps
brachii
long
head
hypertrophy
follow
prom
rt
long
vs
short
muscle
length
long
muscle
length
generally
appear
result
great
degree
passive
tension
passive
tissue
begin
reach
maximal
length
provide
resistance
increase
muscle
length
tension
suggest
activate
mtorc
pathway
associate
muscle
hypertrophy
great
degree
passive
tension
prom
rt
long
musc   �&�M
elbow
extension
show
great
hypertrophy
head
triceps
brachii
long
muscle
length
condition
finding
noteworthy
long
head
triceps
brachii
train
long
muscle
length
o3
hypertro
phy
zone
V#
daugh
ter
�A
conventional
hypertrophy
��5�k
approach
overall
result
suggest
variety
rom
good
effect
injury
management
personal
preference
researcher
interested
see
future
study
compare
adaptation
follow
prom
training
different
muscle
length
compare
example
study
examine
muscle
thickness
adaptation
follow
resistance
training
prom
condition
different
muscle
length
condition
ease
future
analysis
andor
replication
future
research
ensure
datum
openly
available
easy
extract
fail
effort
provide
datum
request
abbreviation
rom
rang   �r�e
error
good
befits
practitioner
range
motion
vs
little
practical
downside
benefit
strategy
small
uncertain
likely
worth
adopt
provide
contraindication
personal
preference
load
availability
injury
management
practitioner
recognize
value
small
effect
existence
relatively
uncertain
small
potential
gain
meaningful
coach
athlete
competitive
recreational
alike
conclusion
outperform
prom
outcome
type
effect
size
range
trivial
small
smds
well
overall
rr
condition
ci
change
favour
appear
small
difference
outcome
depend
exactly
rom
manipulate
eg
short
vs
long
muscle
length
regional
hypertrophy
coachesathlete
wish
adopt
rom
strategy
appropriate
goal
principle
specificity
likely
apply
rom
training
usually
replicate
rom
outcome
interest
approach
good
t  �5
ie
muscle
length
origin
regional
muscle
hypertro
phy
assessment
method
proximal
muscle
hypertrophy
trivial
smd
ci
–
find
favour
dis
tal
muscle
hypertrophy
small
smd
ci
–
find
favour
individual
effect
size
posterior
probability
distribution
overall
subgroup
estimate
find
figure
resistance
training
modality
resistance
training
intervention
categorize
resistance
machine
free
weight
combination
intervention
ex
clusively
resistance
machine
subgroup
analysis
reveal
trivial
smd
ci
�   ?
ie
muscle
length
origin
e;
extend
hip
knee
fully
�U�+
ensure
weight
evenly
distribute
heel
midfoot
perform
barbell
squat
avoid
fall
ѣ �
effect
size
calculate
appropriately
give
study
design
involve
prepost
control
comparison
model
uninformed
prior
recent
metaanalyse
inform
prior
constitute
form
'
double
counting
'
give
study
include
include
likelihood
present
model
model
estimate
monte
carlo
markov
chain
warmup
sampling
iteration
trace
plot
examine
chain
convergence
posterior
predictive
check
examine
model
validity
draw
take
posterior
distribution
construct
probability
density
function
plot
calculate
mean
quantil   
correct
�;
appropriate
technique
 �&O
allow
knee
shift
inward
outward
؝O�
abstractstitle
fulltexts
examine
inclusion
mw
pak
screening
perform
abstrackr
httpabstrackrcebmbrownedu
study
deem
irrelevant
exclude
study
return
search
screen
inclusion
reference
list
include
study
screen
inclusion
publication
cite
include
study
screen
inclusion
quality
assessment
quality
study
meet
inclusion
criterion
assess
testex
scale
testex
scale
alternative
pedro
scale
design
specifically
exercise
science
training
study
show
reliable
compose
item
relate
study
quality
stud   5
absence
change
lbm

T
� ! ��Y ��_ �"�������s�C�	�	�� �������ktcome
eg
muscle
cross
sectional
area
analysis
reveal
trivial
smd
ci
–
favour
finally
a
linear


3
momentary
failure
��G�
set
endpoint
trainee
complete
final
repetition
international
journal
strength
conditioning
phillips
s
m
steele
j
vigotsky
d
possible
repetition
attempt
ed
definitely
achieve
momentary
failure
0}�{
trainee
reach
point
despite
attempt
complete
concentric
por
tion
current
repetition
deviation
prescribe
form
exercise
/�
peri
odization
plan
manipulation
training
variable
order
maximize
training
adaptation
prevent
onset
overtraine
syndrome
	�are   �
transference
hypertrophy
q
p
τ
strength
power
	�  #e
type
outcome
eg
bodyfat
analysis
suggest
trivial
smd
ci
–
favour
figure
display
individual
effect
size
tick
post-
rebound
effect
%
kick
start
P5m
transference
hypertrophy
q
p
τ
strength
power
	�5
training
technique
n-
training
study
5
myogenic
stem
cell
�%
sweet
spot
�b�E
mainly
zdisk
costamere
mechanical
hypertrophy
stimulus
sense
transduce
resistance
exercise
zs�g
metabolite
simply
augment
muscle
activation
cause
mechanotransduction
cascade
large
proportion
muscle
fiber
'5
repeat
bout
effect
�
pump
>
normal

���)
initiate
hypertrophy
stimulus
trigger
hypertrophic
signal
transduction
skeletal
muscle
fiber
hypertrophy
response
resistance
exercise
sensor

rO�
increase
individual
muscle
size
contribute
increase
individual
strength
�%
super
slow
 ؈O�
strategy
important
note
use
rom
necessarily
binary
decision
training
training
prom
analysis
support
hypothesis
perform
prom
rt
long
muscle
length
result
great
muscle
hypertrophy
prom
rt
short
muscle
length
rt
suggest
muscle
hypertrophy
goal
trainee
wish
use
prom
rt
long
muscle
length
training
substantial
support
evidence
concept
resistance
training
long
muscle
length
optimise
hypertrophy
oranchuk
et
al
systematic
review
effect
isometric
training
adaptation
suggest
study
include
i   �;�w
sense
training
perform
alike
test
outcome
favour
prom
group
group
clear
bias
outcome
analysis
reveal
trivial
smd
–
ci
–
favour
prom
outcome
bias
favour
prom
group
trivial
smd
ci
–
favour
outcome
clear
bias
small
smd
ci
favour
outcome
bias
favour
group
individual
effect
size
posterior
prob
ability
distribution
overall
subgroup
estimate
find
figure
muscle
length
muscle
hypertrophy
prom
intervention
categorize
train
muscle
group
hA
sarcoplasmic
hypertrophy
��&�M
rom
regardless
individual
participant
truly
specific
rom
find
table
muscle
length
partial
range
motion
training
worth
note
study
study
examine
prom
perform
–
volume
perform
–
short
muscle
length
contrast
relatively
study
examine
prom
moderate
muscle
length
define
middle
long
muscle
length
specific
finding
study
see
table
grade
table
evidence
see
table
metaanalysis
result
main
model
–
outcome
main
model
–
include
effect
outcome
study
–
reveal
trivial
standardized
mean
difference
smd
ci
–
favour
compare
prom
figure
effect
size
tick
posterior
probability
distribution
overall
estimate
display
figure
subgroup
analysis
group
outcome
type
outcome
group
type
\1
proper
technique
 �'
proper
form
 �!
practice
	(S
outcome
subgroup
analysis
perform
]/
onesize
fitsall
 C
myofibril
expansion
cycle
�'
muscle
size
�m�[
muscle
length
average
assume
muscle
length
important
acknowledge
assumption
joint
angle
muscle
length
likely
correlate
perfectly
purpose
exploratory
subgroup
analysis
assumption
deem
acceptable
prom
condition
low
condition
regard
i��
muscle
length
analysis
reveal
trivial
smd
ci
–
favour
muscle
hypertrophy
prom
perform
short
muscle
length
conversely
prom
perform
long
muscle
length
analysis
show
small
smd
–
ci
–
favour
prom
muscle
hypertrophy
dividual
effect
size
posterior
probability
distribu
tion
overall
subgroup
estimate
find
figure
metaregression
analysis
proportion
set
nonwarmup
set
account
proportion
set
trivial
m
pact
outcome
slope
β
ci
–
quantile
interval
see
figure
figure
muscle
length
prom
su     1 ����������|n_OB4%����������r]O?6' ����������pfZLC3&�����������vl]J?/$�����������zrf\SE:,
�
�
�
�
�
�
�
�
�
�
p
a
R
C
4
$


	�	�	�	�	�	�	�	�	�	�	s	g	Y	N	D	'		����������yh\ND8%���������zpdTE:"���������|i\PC2%����������}k^OB6+����������zjXH@8.$���������rhYOE;0�����������ui]LC1           � %displacement� air� #composition	� assess	� change
�
 measure�	 important
� compare� lbm� mass� lean
� prolong� vs� condition� nogueira	�  keeler� stem	�~ simply�} protocol�| 'heterogeneity�{ topic�z give�y good�x fast�w influence�v wilk�u recently�t #hypothesize�s !previously�r critical
�q finding�p +notwithstanding�o %hypertrophic�n lead�m #combination
�l analyze
�k similar	�j result�i wide�h find�g al�f et�e !schoenfeld�d %metaanalysis�c !systematic	�b tempos�a slow�` #detrimental�_ indicate
�^ support	�] poorly	�\ remain�[ strategy
�Z benefit	�Y manner�X %
super
slow
�W %professional	�V common�U order	�T action
�S combine�R inclusion
�Q usually�P %conventional
�O discuss�N main
�M provide�L approach�K %specifically
�J develop
�I current�H endeavor�G date�F paper�E scholarly�D comprise�C read
�B optimal�A #extrapolate
�@ section	�? method
�> outline�= '
proper
form

�< examine�; study�: previous�9 similarly�8 ambiguity	�7 create	�6 define�5 !explicitly
�4 mention	�3 author�2 infer�1 stage�0 early�/ 1
proper
technique
�. learn�- highlight�, !researcher	�+ depend�* vary�) %prescription�( stand	�' scarce
�& outcome
�% explore�$ directly
�# prevent�" 'effectiveness�! emphasis�  practice� prescribe� alignment� 'orchestration	� injury� risk� minimize	� target� #effectively	� ensure	� bodily� execution
� control
� pertain� !standpoint� power� #potentially� goal� specific� !individual	� affect
� heavily�
 ;
appropriate
technique
�	 note� worth� position	� stance� -widthorientation� grip
� correct
� involve
� primary
�  include checklist	~ manual} nsca| #association{ %conditioningz strengthy nationalx #constituentw !literaturev !scientificu !definitiont !agreedupons #universallyr currently
q proposep differento describen 5
training
technique
m terml effectivek componentj keyi refer
h failureg muscularf momentarye proximity	d effortc intensityb weeka group
` perform_ set^ ie	] volume
\ certain[ %manipulation
Z requireY knowX %introduction	W growthV #developmentU form
T keywordS variation
R lenient	Q versus	P strictO #investigate	N futureM )recommendationL !accordanceK adopt	J followI universal
H suggest
G limited	F effect	E aspect	D impactC #biomechanic
B anatomyA apply	@ theory? imply> base= generally
< pattern; movement: #positioning9 body8 guideline
7 enhance6 phase5 !concentric4 eccentric3 duration2 !manipulate1 determine0 need/ research. s	- length, long	+ employ* recommend) rom	( motion' range& tempo% !repetition$ type# #contraction" kinematic! -exercisespecific  variable focus	 proper !constitute evidence exist !synthesize aim	 review narrative #hypertrophy eg !adaptation maximize try !especially technique exercise #appropriate !importance emphasize %practitioner	
 engage	 size	 muscle increase
 promote show rt training !resistance !regimented   � ��������}pbU;-	�����������ym_S@0%���������xh]TE2%����������ugZPD4%����������wdYMB/$
�
�
�
�
�
�
�
�
�
x
j
\
L
?
6
)


	�	�	�	�	�	�	�	s	g	]	D	7	-			���������}tfWM9$ ���������raOE6'	����������qbQF<3$����������-�������{maXN?3(����������naUH?0 ����������yhYL<+                 
�
 maximum�	 nonstrict� #accordingly� !categorize
� isolate� ancillary� #involvement� place� live� %longitudinal
�  cadaver	� animal�~ human�} #mechanistic�| pain
�{ flexion�z spine	�y lumbar�x reduction�w commonly�v !alteration�u potential�t etc
�s kinetic	�r safety
�q attempt�p %questionable�o #implication�n predictor�m #necessarily�l bench�k pulldown�j lat�i press�h !activation	�g inward
�f outward�e point	�d period�c calf�b alter	�a direct�` objective�_ knee�^ arm	�] moment�\ deadlift�[ close�Z keep�Y reasoning$�X O
allow
knee
shift
inward
outward
	�W adhere�V avoid�U ;
extend
hip
knee
fully
�T %successfully�S complete�R 'jointspecificS�Q �+
ensure
weight
evenly
distribute
heel
midfoot
perform
barbell
squat
avoid
fall
�P like�O correctly�N squat
�M barbell�L #description�K list�J detailed
�I example�H safe�G efficient�F +biomechanically	�E intend�D placement�C bar�B foot�A width�@ #instruction�? principle�> 'biomechanical
�= concept�< 'extrapolation
�; instead�: optimize�9 well�8 majority�7 inception�6 ’�5 !refinement�4 year
�3 product�2 #performance�1 
correct
�0 delineate�/ mean�. %longerlength�- #predisposed�, !understand�+ limit�* -longmusclelength�) 'configuration�( mechanism�' #conceivably�& stop�% add�$ end�# traverse
�" default�! bias
�  publish� +semimembranosus� )semitendinosus� head� composite� extensor� hip� see
� machine� multihip� adductor
� maximus
� gluteus� hamstring� abstract� !conference
� present� #unpublished� asyet� credence� lend
� lateral�
 3shortermusclelength�	 grow	� medial� %longermuscle� 'shortermuscle� )plantarflexion� ankle� 'gastrocnemius� 1longermusclelength� !additional�  #publication� analysis�~ subgroup�} take
�| caution�{ say
�z triceps�y !sufficient�x kassiano�w superior�v wolf�u short	�t distal�s regional�r 'interestingly�q #distinguish�p #dichotomize
�o partial
�n utilize�m #metaanalyse�l !consistent
�k achieve�j large�i 'traditionally�h joint	�g degree�f force�e 'gravitational	�d solely�c rely	�b weight
�a descent�` %sufficiently�_ advisable
�^ unclear�] !acceptable�\ plethora�[ allow�Z occur	�Y appear�X #application�W practical�V !conclusion	�U strong�T draw
�S ability�R preclude�Q #uncertainty�P conflict�O fiber�N iia�M lateralis	�L gillie	�K lastly�J !seccentric�I area�H )crosssectional�G thigh�F squatting�E parallel
�D shibata�C 'alternatively�B extension�A leg�@ train�? !quadriceps�> !marginally
�= pearson�< 'contrastingly�; thickness�: medialis	�9 vastus�8 !experience
�7 overall�6 limb�5 low�4 look	�3 recent�2 extended�1 favor�0 !difference�/ 'statistically	�. biceps�- absolute	�, elicit
�+ pereira�* response�) rtinduced	�( extend�' man�& old	�% cohort
�$ femoris	�# rectus
�" brachii�! bicep�  great� 'significantly� #demonstrate� contrast� 5
absence
change
lbm

� observe� capacity
� aerobic� state� pretopost� #significant� %additionally� !cautiously� interpret� pod� bod� +plethysmography  
( ����������yl_SE6*��������~pe[F4$	�����������qgSH8,!
���������}pdYLB9.#	����������sfXJA6&
�
�
�
�
�
�
�
�
�
�
~
n
`
U
H
=
6
$


	�	�	�	�	�	�	�	�	�	�	�	|	o	b	U	B	7	(		����������reYQG:- ����������xl`RF6%
 ���������rfZH;-�����������vlaVG<+����������se[O<0���������{ndVG5&���������q\QB3(            � angle� plausible� divergent� acute� +mechanistically� %accumulation
� lactate� blood� 'deoxygenation
� dynamic�
 versa�	 vice	� likely� 'anglespecific� 'isometrically� isometric� +transferability� magnitude� %meaningfully� plausibly�  inherent� peak
�~ wingate�} m	�| height�{ +countermovement�z stimulate�y #consolidate�x equivocal	�w report�v #improvement
�u produce�t prom	�s desire
�r context�q #superiority	�p whilst�o 'controversial	�n sprint�m jump�l vertical
�k fieldeg�j one�i strongman�h %powerlifting�g %bodybuilding	�f reward�e #muscularity
�d notably
�c improve	�b induce
�a fullrom�` personal�_ #alternative�^ #efficacious�] speed�\ lowerbody�[ upper�Z clear�Y test�X #specificity�W existence�V small�U smds�T #standardize	�S favour�R ci�Q smd
�P trivial	�O reveal�N model�M moderator�L #exploratory�K !multilevel�J bayesian
�I extract�H %sportsdiscus	�G pubmed	�F prisma�E %registration�D pre
�C bodyfat
�B variety�A )systematically�@ !synthesise	�? differ�> !background
�= college	�< lehman�; york�: new�9 city�8 uk�7 #southampton�6 !university	�5 solent�4 sciences	�3 social	�2 health
�1 faculty	�0 steele�/ j�. brad	�- fisher�, p�+ james�* 5androulakiskorakakis�) patroklos�( milo�' construe�& financial�% !commercial
�$ conduct
�# declare
�" nippard�! jeff�  oa	� strcng
� company� equipment
� fitness� %manufacturer� #corporation� tonal� advisory� serve� interest� %availability� data
� consent	� inform� !applicable� statement� board� 'institutional
� receive
� funding
� version�
 agree�	 !manuscript
� editing
� writing� !contribute� bjs� ap� rb� mc� mw�  idea� conceive�~ pak�} jn�| #nontargeted�{ %contribution�z '�y coach
�x athlete�w sport�v physique�u !perception�t total�s !preference�r think�q flexible
�p subject�o datum
�n paucity
�m reserve�l bent	�k assist�j contrary�i recovery�h !meaningful�g !negligible�f reach�e far�d unlikely�c tradeoff
�b fatigue�a primarily�` work�_ role�^ play�] unwanted�\ inferior�[ yield�Z !unintended�Y generate�X 'strictlenient�W #incorporate�V fit�U #technically�T broad�S table�R –�Q span
�P stretch�O fully	�N design
�M program�L available
�K clarity�J !facilitate�I 'anthropometry	�H relate
�G absence�F standard�E adequate�D maintain�C basis
�B logical�A case�@ truly	�? impose�> !negatively�= argue�< )circumspection�; view�: !simulation�9 row�8 bend�7 spend�6 time�5 decrease�4 #deleterious�3 possible	�2 regard�1 stimulus	�0 impair
�/ loading	�. amount�- excessive�, !conversely�+ %advantageous�* +
well
overload
�) load�( heavy�' moderate�& conclude�% beginning	�$ supply�# raise�" momentum�! external�  %relationship� 1modelingsimulation� !mechanical� %arandjelovic� !indirectly	� albeit� knowledge� addition	� spinae
� erector
� gluteal� !assistance	� permit� forth� sway� drive
� minimal
� posture
� upright� use� curl� #stimulation
- ���G���������������~tgZN@3%���������|o]OB8&!
�
�
�
�
�
�
�
�
u
e
V
J
<
(


	�	�	�	�	�	�	�	�	�	�	u	h	Z	K	=	-	!	�	����������sdVH>4'���������qf]UL':-%�����������whWD2#
����������u�hZM=1'
����������sgTI>3'��sf\Q@0����������}shZOA6-illaranaerobic�abduct.n�Aalphamethylaminoisobutyric�!administer�
agent�amplitude��acidic�0 �arm�
arise	�argument�
argue=arguablya
argua	m	argu�	areaI'architectural�%arandjelovicar�abdominisdabdominusbandersonZabdominalKabductorIadjacent6albumineadrenaldakt   �appreciate+#appreciably
appre�	applyAapplied=#applicationX!applicable�appearYapparent�apoptosis�
apart	�ap�antretter
#antioxidant-antiinflammatory'anthropometryIanterior#answer�
anoth1
ankrd
ankle�'animalderived�animal�'anglespecific
angle#anecdotally�5androulakiskorakakis�androgen�andrew�
andor%ancillary
ancieaanchorage
�anchor
�#ancetrained	ance	�anatomyB!anatomical)analyze �analysisanalyse-+anabolismrelatePanabolism�anabolic�amplify�
ample�ampkαl	ampk�amount.
amotl�'aminoacidemia�
amino�ami�!ameliorate
(ambiguity �%amalgamation�'alternativelyC#alternative�alternate
%!alteration�
alter�alsalongside	�
allow[allcause
_
alitycalignment �
align�alfred
�alexander�aleJalbeitalabama�al �aktmtor%aktmammalianairaimaid
aicariahtiainen�!agreedupont
agree�/agonistantagonist�)agonistagonist�agonist�agef	agar
�)aforementioned�afford�affect �-aerobicendurance	�aerobic	aeroQ	aere�advocate�advisory�advisable_advice�%advantageous+advantagefadvanced�advance�adulthood�
adult�adp↔atp2adp0	adoptK!admittedly�adjust	�adhesion
Xadhere�!adequatelyIadequateEadenosine	�adductor�address
�additive�%additionally!additional�additionadd�adaptive	�!adaptation
adapt�	adap	�ad�acutely$
acuteactuallyiactual
{)activityrelate
�activityWactive1activatorh!activation�activate
�action �#actinmyosind#actinlinkedw/actincytoskeleton
�/actincrosslinkings
actineact
|)acknowledgment�)acidsynthesize(
acids�%acidgenerate
�	acid�achievekaccuracy:%accumulationaccrual	�accretion�
accre�account�#accordingly!accordanceLaccordv!accomplish�accompany�!acceptable]!accentuate"ac�abundanceabstract�abstain,absolute-absent	�absenceGabolish�	ably�	able�ablation
oabilitySab�'z+
well
overload
*#
weak
link
�#
waste
set
`%
vice
versa
j�5�k
variable
betweenstudy
variance
explain
methodological
factor
give
finding
likely
variation
es
study
likely
attribute
sampling
variation
potentially
individual
participant
level
char
acteristic
study
level
charac
teristic
say
comparison
con
dition
variation
treatment
effect
reveal
little
difference
concurrent
single
modality
training
log
variability
ratio
ci
manipulate
hypertrophy
mesocycle
maximize
increase
contractile
tissue
	�N�
type
outcome
eg
sprint
time
analysis
suggest
trivial
smd
ci
–
favour
_E�
type
outcome
eg
rm
test
analysis
reveal
trivial
smd
ci
favour
^R�%
type
outcome
eg
rate
force
development
analysis
show
trivial
smd
ci
favour
`e�K
type
outcome
eg
muscle
cross
sectional
area
analysis
reveal
trivial
smd
ci
–
favour
finally
a   B9 ��������{n^J5%���������yk^H:+����������tcUG9*
���������rfZSH9                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             �V sectional�U cross�T 
	�S string	�R august�Q database�P 'pubmedmedline	�O search�N #restriction�M +groupscondition�L %intervention
�K english	�J thesis	�I master�H doctoral�G %peerreviewed�F fulltext�E criterion�D original�C template�B prospero�A #prospective�@ 'international�? %httpsosfioje�> osf�= framework
�< science�; open�: register�9 item�8 reporting�7 preferred�6 !morphology
�5 article�4 !subanalyse�3 multitude�2 totality�1 -metaanalytically
�0 despite�/ scarcity�. upperlimb
�- analyse
�, abstain�+ lowerlimb�* !functional
�) pallare�( #musculature�' grgic�& function�% andor�$ 'morphological�# summarize�" !accentuate�! +musculoskeletal�  )athletetrainee� !beneficial
� finally� pitching� baseball� task	� moreso� scrum� rugby� )sportsmovement� /
onesize
fitsall
� %taskspecific
   �� ��{l^R�B1$���������~lYQOE:,!�������������{l`QA1����������vkbTG;+���������|dTE,������������sf=\SF;2$ ������G�����rbUKB0
�
�
�
�
�
�
�
�
�
�
~
o
_
L
7
'

	�	�	�	�	�	�	�	��	�	�	�	�	w	f	T�	?	2	"		�����co�����yncXK@4'#�������������yl]G<2(����������vdWK:.�������/����������|qfYNF5* 
coachy#cndependent+cn(
cloudclosely
close�
climb�cleverlcleotide4clearly�clearance�
clear�cle-classic	�clarityK
cadence|blunting{burnoutybureshq
bodyspbufferncategoryg	balla	behm[%architectureWbranch=)branchsuggeste9	bear'applicabilitybyproductcapillary artery�/bodybuildingstyle�%artificially�arouse�arm�
arise	�argument�
argue=arguablya
argua	m	argu�	areaI'architectural�%arandjelovicar�aqueousDaquaporin�aptation�'approximatelyapprove�#appropriateapproach �!appreciate+#appreciably
appre�	applyAapplied=#applicationX!applicable�appearYapparent�apoptosis�
apart	�ap�antretter
causal;!categorizecatalyze�catabolic#castinduced
�	cast�	caseAcascade
C	casa�
carry�#caregulated)career	�#cardiotoxin�cardio	�cardiac�%carbohydrate�car	�capture%capacitycapable
�!capability capicandidate
Kcancer
�canadian
=canada�camkiv�camkii�!calmodulin�
cally�	callV	calf�
calcu-calciumdependent$calciumcacalciumd#calcineurin'cal	CcaffeineF#cadependent%cadaver 
cable�c�butanol%bundleO	bulk�buildup�
build�buford	�buffering�buckner�brooksI
bronx�broadly�
broadT
bringF
brilso	bril�briefly?
briefbreakdown�
break�brandao�	brad�brachii"+braceimmobilizebovine�	bout�	bone
�)bodyweightmeal�!bodyweightbodyfat�+bodycomposition�%bodybuilding�#bodybuilder�body9bodily �bod
board�bly�
blunt�!bloodbased	d
bloodblockade
m
block�bjs�bj�%bisphosphate!birmingham�biopsy;!biomedical�+biomechanically�'biomechanical�#biomechanicCbiomarker4%biologicallyi!biological"binding`	bind_binary�bin�big�biceps.
bicep!bicRbiasedg	bias�bi	"bfr�)betweensubject<%betweengroup�ber�	bentlbenefit �!beneficialbeneath;	bend8
bench�benJbelieveJbelief�
being
�beginning%
begin�bed�bazvallebayesian�
basisC
basic�baseline>baseballbase>
basal�barbell�bar�	band	balance�!bagfocused�bag�!background�b�	axisb
axial�ax�	away�
avoid�
avian�average{availableL%availability�
avail�'autoregulated3autophosphorylation
�autophagy
O!autophaghy�1autocrineparacrine�autocrineRauthor �australia�augustRaugmentsauckland�
ature�attribute�attract�attenuate(attention(attempt�attain	~!attachment
�attachxatrophy
�atpbinde\atp^
ativeGationally�ational
ationr	aticathleticv)athletetrainee athletexath�atetohigh�ate
	
asyet�!assumption	�assumeM'associational�#association|!associated
�associate�associassistant�!assistanceassistkassignh!assessment	assessaspire�
aspectEarticle5arthur	3#arrangement�    y
report
finally
grade
table
evidence
table
produce
clearly
communicate
finding
gradepro
httpswwwgradeproorg
perform
mw
data
extraction
follow
datum
extractedcode
study
meet
inclusion
criterion
mw
study
design
weight
mean
age
weight
mean
height
intervention
duration
total
study
duration
sex
participant
train
status
population
rom
prom
groupcondition
rom
groupcondition
proportion
set
perform
fullprom
muscle
length
train
training
frequency
mean
number
weekly
set
perform
mean
repetition
duration
mean
number
repetition
perform
set
number
exercise
mean
proximity
momentary
muscular
failure
mean
load
modality
training
presence
auxiliary
intervention
exercise
perform
exercise
rom
manipulate
preregistration
note
grouping
outcome
musculoskeletal
function
morphology
note
systematic
search
review
additional
outcome
extract
depend
study
measure
datum
extraction
opt
group
outcome
follow
category
body
composition
outcome
strength
outcome
power
outcome
sport
outcome
finally
outcome
measure
favour
bias
prom
groupcondition
eg
partial
squat
repetition
maximum
rm
favour
partial
squat
group
note
datum
available
fulltext
author
contact
request
miss
datum
contact
information
unavailable
institution
work
perform
contact
obtain
response
receive
initial
request
second
email
send
week
later
response
obtain
second
attempt
datum
obtain
webplotdigitizer
v
ankit
rohatgi
possible
datum
transcribedimporte
csv
file
metaanalysis
analysis
code
utilize
present
supplementary
materials
httpsosfiofmvrw
give
aim
research
opt
estimationbased
approach
conduct
bayesian
framework
analysis
effect
estimate
precision
conclusion
base
interpret
continuously
probabilistically
consider
datum
quality
plausibility
effect
previous
literature
context
outcome
main
exploratory
metaanalysis
perform
'
brm
'
package
posterior
draw
take
'
tidybayes
'
'
emmean
'
r
v
r
core
team
httpswwwrprojectorg
datum
visualization
'
ggplot
'
'
patchwork
'
include
study
multiple
group
condition
report
effect
multiple
sessionsexercisesset
opt
calculate
effect
size
nested
structure
multilevel
mixedeffect
metaanalyse
perform
interstudy
intra
study
group
include
random
effect
model
effect
weight
inverse
sampling
variance
account
study
variance
main
model
include
effect
outcome
include
study
conduct
exploratory
metaregression
subgroup
analysis
moderator
ie
predictor
effect
explore
study
protocol
participant
characteristic
moderator
examine
include
international
journal
strength
conditioning
review
metaanalysis
outcome
subcategory
strength
muscle
size
body
fat
power
sport
performance
proxy
study
design
withinparticipant
upper
vs
low
body
length
muscle
train
prom
condition
short
middle
long
specifically
explore
muscle
size
outcome
modality
resistance
free
weight
resistance
machine
combination
outcome
measure
way
specifically
bias
prom
eg
rm
outcome
bias
vice
versa
prom
rm
outcome
prom
participant
’
mean
height
consider
relate
limb
length
intervention
duration
proportion
volume
perform
proportion
prom
condition
time
load
repetition
muscle
size
proximal
distal
muscle
site
measure
primary
model
produce
standardised
mean
difference
smd
ie
hedge
g
effect
size
present
present
supplementary
model
log
response
ratio
mean
rr
exponentiate
present
percentage
difference
change
prom
available
supplementary
file
httpsosfiofmvrw
folder
W                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          /#                                                                                                                                                                                                                                                             �Z lnrr	�Y output	�X figure�M�W �
abstractstitle
fulltexts
examine
inclusion
mw
pak
screening
perform
abstrackr
httpabstrackrcebmbrownedu
study
deem
irrelevant
exclude
study
return
search
screen
inclusion
reference
list
include
study
screen
inclusion
publication
cite
include
study
screen
inclusion
quality
assessment
quality
study
meet
inclusion
criterion
assess
testex
scale
testex
scale
alternative
pedro
scale
design
specifically
exercise
science
training
study
show
reliable
compose
item
relate
study
quality
study
report
finally
grade
table
evidence
table
produce
clearly
communicate
finding
gradepro
httpswwwgradeproorg
perform
mw
data
extraction
follow
datum
extractedcode
study
meet
inclusion
criterion
mw
study
design
weight
mean
age
weight
mean
height
intervention
duration
total
study
duration
sex
participant
train
status
population
rom
prom
groupcondition
rom
groupcondition
proportion
set
perform
fullprom
muscle
length
train
training
frequency
mean
number
weekly
set
perform
mean
repetition
duration
mean
number
repetition
perform
set
number
exercise
mean
proximity
momentary
muscular
failure
mean
load
modality
training
presence
auxiliary
intervention
exercise
perform
exercise
rom
manipulate
preregistration
note
grouping
outcome
musculoskeletal
function
morphology
note
systematic
search
review
additional
outcome
extract
depend
study
measure
datum
extraction
opt
group
outcome
follow
category
body
composition
outcome
strength
outcome
power
outcome
sport
outcome
finally
outcome
measure
favour
bias
prom
groupcondition
eg
partial
squat
repetition
maximum
rm
favour
partial
squat
group
note
datum
available
fulltext
author
contact
request
miss
datum
contact
information
unavailable
institution
work
perform
contact
obtain
response
receive
initial
request
second
email
send
week
later
response
obtain
second
attempt
datum
obtain
webplotdigitizer
v
ankit
rohatgi
possible
datum
transcribedimporte
csv
file
metaanalysis
analysis
code
utilize
present
supplementary
materials
httpsosfiofmvrw
give
aim
research
opt
estimationbased
approach
conduct
bayesian
framework
analysis
effect
estimate
precision
conclusion
base
interpret
continuously
probabilistically
consider
datum
quality
plausibility
effect
previous
literature
context
outcome
main
exploratory
metaanalysis
perform
'
brm
'
package
posterior
draw
take
'
tidybayes
'
'
emmean
'
r
v
r
core
team
httpswwwrprojectorg
datum
visualization
'
ggplot
'
'
patchwork
'
include
study
multiple
group
condition
report
effect
multiple
sessionsexercisesset
opt
calculate
effect
size
nested
structure
multilevel
mixedeffect
metaanalyse
perform
interstudy
intra
study
group
include
random
effect
model
effect
weight
inverse
sampling
variance
account
study
variance
main
model
include
effect
outcome
include
study
conduct
exploratory
metaregression
subgroup
analysis
moderator
ie
predictor
effect
explore
study
protocol
participant
characteristic
moderator
examine
include
international
journal
strength
conditioning
review
metaanalysis
outcome
subcategory
strength
muscle
size
body
fat
power
sport
performance
proxy
study
design
withinparticipant
upper
vs
low
body
length
muscle
train
prom
condition
short
middle
long
specifically
explore
muscle
size
outcome
modality
resistance
free
weight
resistance
machine
combination
outcome
measure
way
specifically
bias
prom
eg
rm
outcome
bias
vice
versa
prom
rm
outcome
prom
participant
’
mean
height
consider
relate
limb
length
intervention
duration
proportion
volume
perform
proportion
prom
condition
time
load
repetition
muscle
size
proximal
distal
muscle
site
measure
primary
model
produce
standardised
mean
difference
smd
ie
hedge
g
effect
size
present
present
supplementary
model
log
response
ratio
mean
rr
exponentiate
present
percentage
difference
change
prom
available
supplementary
file
httpsosfiofmvrw
folder
    e
interval
'
credible
'
'
compatibility
'
interval
posterior
probability
density
function
group
effect
estimate
give
probable
value
parameter
give
level
probability
search
result
search
string
identify
publication
thesis
potential
inclusion
identify
website
citation
search
duplicate
remove
study
remain
title
abstract
screen
deem
appropriate
fulltext
version
seek
determine
eligibility
ultimately
study
include
review
study
eventually
exclude
data
extraction
excessive
miss
datum
thesis
exclude
contain
datum
publication
include
figure
detail
process
table
provide
summary
datum
study
include
analysis
c
c
number
core
available
computer
run
analysis
build
available
https
ukpcpartpickercomlistcvxrt
figure
prisma
flow
chart
table
summary
study
include
rhea
et
al
participant
group
l
°
°
significant
difference
favour
rm
prom
prom
rm
vert
jump
sprint
test
valamatos
et
al
goto
et
al
esmaeeldokht
martinezcava
et
al
kubo
et
al
partici
pant
par
ticipant
par
ticipant
par
ticipant
par
ticipant
l
°
°
significant
difference
muscle
size
maximum
torque
improve
significantly
respective
rom
u
°
°
significantly
great
improvement
muscle
csa
iso
metric
strength
prom
l
significant
betweengroup
difference
rms
body
fat
u
rom
generally
lead
well
rm
mpv
rm
outcome
strength
gain
great
train
rom
l
°
°
significantly
great
improvement
rm
adduc
torgluteus
maximus
growth
prom
pallare
et
al
whaley
et
al
participant
group
par
l
°
rom
generally
lead
well
rm
mpv
rm
outcome
strength
gain
great
train
rom
significant
difference
wgt
cmjsprint
time
similar
improvement
vj
height
squat
rm
ticipant
l
power
output
increase
rom
prom
compare
continuously
train
sadacharan
seo
werkhausen
et
al
pedrosa
et
al
partici
pant
partici
pant
partici
°
prom
generally
lead
improvement
mvic
l
°
°
prom
generally
lead
similar
improvement
peak
torque
force
power
rtd
muscle
thickness
train
prom
long
muscle
length
generally
result
pant
group
l
°
°
great
muscular
strength
adaptation
prom
short
muscle
length
rom
assume
base
exist
biomechanical
analysis
squat
depth
detail
available
supplementary
material
rom
digitize
manuscript
detail
available
supplementary
material
table
grade
table
evidence
muscle
strength
followup
median
week
assess
isometric
strength
isometric
torque
partial
rom
rm
rom
rm
relative
peak
force
rm
peak
force
maximum
voluntary
contraction
specific
tension
fascicle
force
specific
torque
relative
rom
rm
relative
partial
rom
rm
run
domise
trial
seri
ous
seriou
sa
smd
sd
high
low
higherb
moderate
alt
text
sport
followup
median
week
assess
stand
vertical
jump
height
depth
jump
height
countermovement
jump
vertical
velocity
countermovement
jump
height
countermovement
jump
force
squat
jump
height
yard
sprint
time
meter
sprint
time
power
followup
median
week
assess
relative
peak
power
countermovement
height
countermovement
force
halfrom
force
unilateral
maximal
rate
force
development
isometric
rate
force
development
mean
propulsive
velocity
different
rm
rom
peak
mean
power
wingate
test
peak
power
peak
velocity
run
domise
trial
seri
ous
seriou
sa
smd
sd
high
high
high
moderate
muscle
size
followup
median
week
assess
muscle
thickness
regional
crosssectional
area
muscle
volume
smd
run
domise
trial
seri
ous
seriou
sa
sd
high
low
high
moderate
alt
text
body
fat
followup
median
week
assess
body
fat
percentage
regional
subcutaneous
fat
thickness
skinfold
body
fat
waist
hip
ratio
ci
confidence
interval
smd
standardised
mean
difference
explanation
smds
large
datum
unavailable
request
study
b
smd
summary
study
characteristic
range
motion
control
method
control
rom
varied
study
tostudy
study
mechanical
stop
build
equipment
–
isokinetic
dynamometerselectric
goniometerstensiometer
study
participant
'
rom
control
physical
stop
like
metallic
bar
delineate
partial
rom
pedrosa
et
al
pinto
et
al
finally
study
rom
clearly
define
participant
supervise
personnel
ensure
rom
correct
–
accuracy
method
ideal
interesting
note
study
individualise
rom
individual
word
study
certain
rom
deem
[                                                                                                         e
interval
'
credible
'
'
compatibility
'
interval
posterior
probability
density
function
group
effect
estimate
give
probable
value
parameter
give
level
probability
search
result
search
string
identify
publication
thesis
potential
inclusion
identify
website
citation
search
duplicate
remove
study
remain
title
abstract
screen
deem
appropriate
fulltext
version
seek
determine
eligibility
ultimately
study
include
review
study
eventually
exclude
data
extraction
excessive
miss
datum
thesis
exclude
contain
datum
publication
include
figure
detail
process
table
provide
summary
datum
study
include
analysis
c
c
number
core
available
computer
run
analysis
build
available
https
ukpcpartpickercomlistcvxrt
figure
prisma
flow
chart
table
summary
study
include
rhea
et
al
participant
group
l
°
°
significant
difference
favour
rm
prom
prom
rm
vert
jump
sprint
test
valamatos
et
al
goto
et
al
esmaeeldokht
martinezcava
et
al
kubo
et
al
partici
pant
par
ticipant
par
ticipant
par
ticipant
par
ticipant
l
°
°
significant
difference
muscle
size
maximum
torque
improve
significantly
respective
rom
u
°
°
significantly
great
improvement
muscle
csa
iso
metric
strength
prom
l
significant
betweengroup
difference
rms
body
fat
u
rom
generally
lead
well
rm
mpv
rm
outcome
strength
gain
great
train
rom
l
°
°
significantly
great
improvement
rm
adduc
torgluteus
maximus
growth
prom
pallare
et
al
whaley
et
al
participant
group
par
l
°
rom
generally
lead
well
rm
mpv
rm
outcome
strength
gain
great
train
rom
significant
difference
wgt
cmjsprint
time
similar
improvement
vj
height
squat
rm
ticipant
l
power
output
increase
rom
prom
compare
continuously
train
sadacharan
seo
werkhausen
et
al
pedrosa
et
al
partici
pant
partici
pant
partici
°
prom
generally
lead
improvement
mvic
l
°
°
prom
generally
lead
similar
improvement
peak
torque
force
power
rtd
muscle
thickness
train
prom
long
muscle
length
generally
result
pant
group
l
°
°
great
muscular
strength
adaptation
prom
short
muscle
length
rom
assume
base
exist
biomechanical
analysis
squat
depth
detail
available
supplementary
material
rom
digitize
manuscript
detail
available
supplementary
material
table
grade
table
evidence
muscle
strength
followup
median
week
assess
isometric
strength
isometric
torque
partial
rom
rm
rom
rm
relative
peak
force
rm
peak
force
maximum
voluntary
contraction
specific
tension
fascicle
force
specific
torque
relative
rom
rm
relative
partial
rom
rm
run
domise
trial
seri
ous
seriou
sa
smd
sd
high
low
higherb
moderate
alt
text
sport
followup
median
week
assess
stand
vertical
jump
height
depth
jump
height
countermovement
jump
vertical
velocity
countermovement
jump
height
countermovement
jump
force
squat
jump
height
yard
sprint
time
meter
sprint
time
power
followup
median
week
assess
relative
peak
power
countermovement
height
countermovement
force
halfrom
force
unilateral
maximal
rate
force
development
isometric
rate
force
development
mean
propulsive
velocity
different
rm
rom
peak
mean
power
wingate
test
peak
power
peak
velocity
run
domise
trial
seri
ous
seriou
sa
smd
sd
high
high
high
moderate
muscle
size
followup
median
week
assess
muscle
thickness
regional
crosssectional
area
muscle
volume
smd
run
domise
trial
seri
ous
seriou
sa
sd
high
low
high
moderate
alt
text
body
fat
followup
median
week
assess
body
fat
percentage
regional
subcutaneous
fat
thickness
skinfold
body
fat
waist
hip
ratio
ci
confidence
interval
smd
standardised
mean
difference
explanation
smds
large
datum
unavailable
request
study
b
smd
summary
study
characteristic
range
motion
control
method
control
rom
varied
study
tostudy
study
mechanical
stop
build
equipment
–
isokinetic
dynamometerselectric
goniometerstensiometer
study
participant
'
rom
control
physical
stop
like
metallic
bar
delineate
partial
rom
pedrosa
et
al
pinto
et
al
finally
study
rom
clearly
define
participant
supervise
personnel
ensure
rom
correct
–
accuracy
method
ideal
interesting
note
study
individualise
rom
individual
word
study
certain
rom
deem
                                                                                                          � g>
�
�
V	�	������                                                                                                                                                                                                                                                                                                                                                                                                                                                            	�g biased�X�f �5
ie
muscle
length
origin
regional
muscle
hypertro
phy
assessment
method
proximal
muscle
hypertrophy
trivial
smd
ci
–
find
favour
dis
tal
muscle
hypertrophy
small
smd
ci
–
find
favour
individual
effect
size
posterior
probability
distribution
overall
subgroup
estimate
find
figure
resistance
training
modality
resistance
training
intervention
categorize
resistance
machine
free
weight
combination
intervention
ex
clusively
resistance
machine
subgroup
analysis
reveal
trivial
smd
ci
–
favour
intervention
exclusively
free
weight
analysis
show
trivial
smd
ci
–
favour
finally
intervention
combination
mo
dalitie
analysis
reveal
small
smd
ci
–
favour
individual
effect
size
posterior
probability
distribution
overall
subgroup
estimate
find
figure
upper
vs
low
body
study
group
training
low
upperbody
upperbody
intervention
analysis
show
trivial
smd
ci
–
favour
likewise
lowerbody
tervention
analysis
reveal
trivial
smd
ci
–
favour
individu
al
effect
size
posterior
probability
distribution
overall
subgroup
estimate
find
figure
figure
regional
hypertrophy
subgroup
analysis
figure
resistance
modality
subgroup
analysis
figure
upper
vs
low
body
subgroup
analysis
figure
outcome
bias
subgroup
analysis
outcome
bias
outcome
group
�e ?
ie
muscle
length
origin
�d proximal�p�c �e
type
outcome
eg
bodyfat
analysis
suggest
trivial
smd
ci
–
favour
figure
display
individual
effect
size
tick
posterior
probability
distribution
overall
estimate
outcome
study
design
study
categorize
subject
design
eg
subject
different
range
motion
different
limb
betweensubject
design
eg
subject
assign
perform
intervention
prom
intervention
withinparticipant
design
analysis
reveal
small
smd
ci
–
favour
outcome
participant
design
analysis
reveal
trivial
smd
ci
–
favour
figure
display
individual
effect
size
tick
posterior
probability
distribution
overall
estimate
outcome
figure
overall
model
figure
outcome
subgroup
analysis
figure
study
design
subgroup
analysis
proximal
vs
distal
muscle
hypertrophy
hypertrophy
outcome
assessment
group
�b fatc�a �K
type
outcome
eg
muscle
cross
sectional
area
analysis
reveal
trivial
smd
ci
–
favour
finally
P�` �%
type
outcome
eg
rate
force
development
analysis
show
trivial
smd
ci
favour
L�_ �
type
outcome
eg
sprint
time
analysis
suggest
trivial
smd
ci
–
favour
C�^ �
type
outcome
eg
rm
test
analysis
reveal
trivial
smd
ci
favour
&�] S
outcome
subgroup
analysis
perform
�$�\ �M
rom
regardless
individual
participant
truly
specific
rom
find
table
muscle
length
partial
range
motion
training
worth
note
study
study
examine
prom
perform
–
volume
perform
–
short
muscle
length
contrast
relatively
study
examine
prom
moderate
muscle
length
define
middle
long
muscle
length
specific
finding
study
see
table
grade
table
evidence
see
table
metaanalysis
result
main
model
–
outcome
main
model
–
include
effect
outcome
study
–
reveal
trivial
standardized
mean
difference
smd
ci
–
favour
compare
prom
figure
effect
size
tick
posterior
probability
distribution
overall
estimate
display
figure
subgroup
analysis
group
outcome
type
outcome
group
type
�~�[ �
effect
size
calculate
appropriately
give
study
design
involve
prepost
control
comparison
model
uninformed
prior
recent
metaanalyse
inform
prior
constitute
form
'
double
counting
'
give
study
include
include
likelihood
present
model
model
estimate
monte
carlo
markov
chain
warmup
sampling
iteration
trace
plot
examine
chain
convergence
posterior
predictive
check
examine
model
validity
draw
take
posterior
distribution
construct
probability
density
function
plot
calculate
mean
quantil   
9 �'gWKC5+�D�����������tcXL?(��������r�j^QF;2#���������k��wj\PE:-"	���������yk\O?- ����������~mZM={*
��*�������~pScTC0"�
�
��
�
�
�
�
�
�
�
v
c
T0
@
3
 

	�	�	�'�	�	�	�:	�	�	�	q	a	T	H	8	-		
�����������wfTB."(��������t����������{mbXLD9����������u_�D���ujaPB:                        #customarily�cytokine,%conclusively�� �cycle�cyccustom	F	cuse
*
curve�currentlyrcurrent �	curl	cupyt	cuny�!cumulative�!cumferencecultured
�cultureGculprit	�culminate.
cular�
csapo	�
cruitm
crude
cru)concentricallytcentrallyC1componentsdistinct8!clavicular2#compartment)cognizant%#chronicallycessation!completion'consecutively!compressed�!coactivate
coachy#cndependent+cn(
cloudclosely
close�
climb�cleverlcleotide4clearly�clearance�
clear�cle-classify�classic	�clarityK
claim�	cjun 	city�citrateRcising�
cises�	cise�)circumspection<circulateFcir 
cific�
cient	�cialize�ciable�ci�chronic�#chromosomalpchoose1cholesterolderivedZchoiceg!chillibeck
chest�	chen�checklist%characterize)characteristic	�character	�/chaperoneassisted�channel�change
chang#challenging�challenge[
chain�certainly�certain\cer�	cept 	ceps	#century�centuate	Rcentric	8!centration�cellularB#celldeplete�cellbaseZ	cell�ceivablycc
�caveat
�!cautiouslycaution|!constraint!constitute#constituentxconstant+!consortium�#consolidate�%consistently�!consistentlconsistJ'consideration�consider�consid�)conservatively�%conservative�%consequentlyh#consequence�consent�consensus�!connective2!connection�connect
�#confounding
�!confounder:confoundOconflictPconfine�'configuration�!confidencep
confiG!conference�conferconduct�!conditions�%conditioning{condition
condi	�condensed
0%concurrently	�!concurrent	�concurO!conclusive�!conclusionVconclude&conclu	�concern	�'conceptualize	concept�)concentriconly	!concentric5'concentration�concen	conceive#conceivably�#conceivable�conceiva�conceiv�conZcomputer
comprise �#compression
�'comprehensive(compre
compoundt'compositional�#compositioncomposite�compose3componentk
compoi#complicated�!complicate�complex!completely�complete�'complementary�#competitiveX#competition�!compensate!compelling�'compartmental8!comparisonXcompare'comparatively�#comparative�!comparable�company�commonly�common �!commercial�	come�combine �#combination �combina�
combi	�com
%colloquially	�college�%collectively!collection+collectQcolleague�collagen4colW3coimmunoprecipitate�cofactorcoexpress coupling�#contributor�#conjunction�!compromisewcommitt)conformationaln: cytoplasmm%considerable[ �dedaysweek	#daystoweeksdayW
datumo	date �)datageneratingLdatabaseQ	data�'damagerelated�damaged0	codyzcohort%cohesive,on#concomitant�D (count�#contractile�couple�!competitor�	cond�   �damageinjuryassociate
I-damageassociated�damagedality�
dairy
5
daily
&da%d�cγcytosol�%cytoskeleton
�%cytoskeletalM)cyclooxygenase    ��
favour
intervention
exclusively
free
weight
analysis
show
trivial
smd
ci
–
favour
finally
intervention
combination
mo
dalitie
analysis
reveal
small
smd
ci
–
favour
individual
effect
size
posterior
probability
distribution
overall
subgroup
estimate
find
figure
upper
vs
low
body
study
group
training
low
upperbody
upperbody
intervention
analysis
show
trivial
smd
ci
–
favour
likewise
lowerbody
tervention
analysis
reveal
trivial
smd
ci
–
favour
individu
al
effect
size
posterior
probability
distribution
overall
subgroup
estimate
find
figure
figure
regional
hypertrophy
subgroup
analysis
figure
resistance
modality
subgroup
analysis
figure
upper
vs
low
body
subgroup
analysis
figure
outcome
bias
subgroup
analysis
outcome
bias
outcome
group
f                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
   �: ���������vfTE7&��������pdWD2"����������teTD1 �������Z�}obVF;�/#���������}rgn\QB5(

�
�
�
�
�
�
�
�
�
�
�
|
h
Z
K
9
'




	�	�	�	�	�	�	�	�	y	k	W	M	B	9	(			����������vl`RC7&���������xla�UE4)M
�������:���veYJ>2%���������}nbVI>*����������l[LAz8,�����                                                                                                                                                                                                                                                                                                                                                                      'derecruitment�densitym)counterbalanceicrunch_diminish\coverageNdeltoid3depress/cortisol discover
�!discomfort�!disclosure�#discernible	y%disassociateldirectly �direction
~direct�dio	�
dimer�difficult�#differently+differentiation)differentiatedL'differentiateTdifferentp!difference0differ�difQdietary�	diet
idictate�#dichotomizepdiatorAdiameter�)diacylglycerol,
dgkξ.devoted�devote�device	deviation	�#developmentVdevelop �
devel3deuteriumOdeuterate#detrimental �detriment	�detrimen�determine1detect�detailed�detail*despite0desmin�desire�desirable�designN#description�describeodescentadescend	derive�+dephosphorylate�dependentd!dependencedepend �#departments�!department�'deoxygenationdenote
denceH#demonstrate!demonstrat�demand�
delta	�deload
"delivery6deliver�delineate�deline
#deleterious4
delaydegreeg#degradation�/deformingyapmtorc�deformed}5deformationinitiated
T#deformation
�deformv!definitionudefiniteCdefine �	defi�default�decrement�decrease5declinedeclare�decipher�decidesdecade	�debris�debate�dearth	kdeadlift�dedaysweek	#daystoweeksdayW
datumo	date �)datageneratingLdatabaseQ	data�'damagerelated�7damageinjuryassociate
Idamaged0-damageassociated�damagedality�
dairy
5
daily
&da%d�cγcytosol�%cytoskeleton
�%cytoskeletalMcytoplasmmcytokine,)cyclooxygenase
cycle�cyc#customarily�custom	F	cuse
*
curve�currentlyrcurrent �	curl	cupyt	cuny�!cumulative�!cumferencecultured
�cultureGculprit	�culminate.
cular�
csapo	�
cruitm
crude
crual�)crosssectionalHcrosssec	
crossUcritical �criterionE	crit}credulity�credence�creatine�create �!creasingly�crease�covery�
cover!covariance)course.courage&coupling�couple�+countermovement�
count�council
6!costameres
U+costamererelate
�'costamerebase<1costamereassociate
�costamere
�
costa�#corroborate	�'correspondingo!correspond�#correlation correlatecorrela[correctly�correct �
corre5#corporation�	core	cord	Ncor+coplasm�%coordinationcooccur�!convincingk!conversion !conversely,%conventional �'controversial�control �contro=%contributory_#contributor�%contribution{!contribute�contribu�'contrastingly<contrastcontraryj'contradictory�#contraction##contracting�#contractile�contractucontinuum�%continuously/!continuous�continue	P#continually]contin	^context�!contention�content�contain]consume
hconsult'construe�    � CTB �                                                                                                                                                                  ��k �
muscle
length
analysis
reveal
trivial
smd
ci
–
favour
muscle
hypertrophy
prom
perform
short
muscle
length
conversely
prom
perform
long
muscle
length
analysis
show
small
smd
–
ci
–
favour
prom
muscle
hypertrophy
dividual
effect
size
posterior
probability
distribu
tion
overall
subgroup
estimate
find
figure
metaregression
analysis
proportion
set
nonwarmup
set
account
proportion
set
trivial
m
pact
outcome
slope
β
ci
–
quantile
interval
see
figure
figure
muscle
length
prom
subgroup
analysis
figure
proportion
volume
rom
metaregression
international
journal
strength
conditioning
steele
j
proportion
prom
condition
proportion
prom
condi
tion
trivial
impact
outcome
slope
β
ci
–
quantile
interval
see
figure
height
height
participant
trivial
impact
come
slope
β
ci
–
quantile
interval
see
figure
figure
proportion
prom
train
metaregression
figure
participant
height
metaregression
intervention
duration
duration
training
intervention
trivial
impact
outcome
slope
β
–
ci
–
quantile
interval
see
figure
time
load
time
load
repetition
trivial
m
pact
outcome
slope
β
–
ci
–
quantile
interval
see
figure
figure
intervention
duration
metaregression
figure
participant
height
metaregression
quality
assessment
quality
evidence
testex
scale
assess
study
quality
see
table
range
testex
score
commonly
meet
criterion
study
quality
include
group
similar
baseline
titrationprogression
relative
training
intensity
program
statistical
test
’
result
report
commonly
meet
criterion
include
complete
reporting
outcome
datum
include
measure
variance
point
estimate
measure
andor
report
adherence
intervention
potential
bias
review
process
review
unique
feature
inclusion
master’sdoctoral
thesis
include
theses
datum
analyse
great
confidence
finding
review
review
screen
abstract
separate
database
addition
referencecitation
checking
hope
entirety
literature
rom
vast
majority
relevant
literature
include
inclusion
criterion
purposely
keep
simple
lenient
reason
use
testex
scale
provide
gauge
study
quality
say
review
suffer
meaningful
limitation
firstly
inclusion
thesis
result
inclusion
datum
undergo
peer
review
process
rigorous
publish
datum
secondly
effort
manuscript
indicate
subgroup
regression
analysis
deem
exploratory
worth
reiterate
analysis
lack
datum
statistical
power
confident
inference
finally
effort
obtain
datum
possible
unable
obtain
datum
possible
result
review
meaningfully
different
datum
available
discussion
article
aim
review
metaanalyse
effect
rom
rt
range
outcome
major
finding
systematic
review
metaanalysis
rom
rt
appear
modest
impact
outcome
interest
outcome
pool
impact
rom
trivial
small
example
rr
range
show
difference
condition
ci
change
favour
httpsosfioahjnf
result
suggest
different
rom
appropriate
different
goal
example
train
specific
performance
outcome
eg
partial
squat
rm
powerlifting
competition
appear
training
similar
rom
maximise
improvement
trivial
small
margin
result
strongly
suggest
principle
specificity
apply
rom
–
benefit
modest
commonly
assume
look
outcome
group
category
eg
muscle
size
strength
difference
result
prom
largely
trivial
say
noteworthy
effect
size
small
magnitude
directionally
favour
utilise
resistance
training
prove
effective
�j %
vice
versa
�k�i �[
muscle
length
average
assume
muscle
length
important
acknowledge
assumption
joint
angle
muscle
length
likely
correlate
perfectly
purpose
exploratory
subgroup
analysis
assumption
deem
acceptable
prom
condition
low
condition
regard
�9�h �w
sense
training
perform
alike
test
outcome
favour
prom
group
group
clear
bias
outcome
analysis
reveal
trivial
smd
–
ci
–
favour
prom
outcome
bias
favour
prom
group
trivial
smd
ci
–
favour
outcome
clear
bias
small
smd
ci
favour
outcome
bias
favour
group
individual
effect
size
posterior
prob
ability
distribution
overall
subgroup
estimate
find
figure
muscle
length
muscle
hypertrophy
prom
intervention
categorize
train
muscle
group
    bgroup
analysis
figure
proportion
volume
rom
metaregression
international
journal
strength
conditioning
steele
j
proportion
prom
condition
proportion
prom
condi
tion
trivial
impact
outcome
slope
β
ci
–
quantile
interval
see
figure
height
height
participant
trivial
impact
come
slope
β
ci
–
quantile
interval
see
figure
figure
proportion
prom
train
metaregression
figure
participant
height
metaregression
intervention
duration
duration
training
intervention
trivial
impact
outcome
slope
β
–
ci
–
quantile
interval
see
figure
time
load
time
load
repetition
trivial
m
pact
outcome
slope
β
–
ci
–
quantile
interval
see
figure
figure
intervention
duration
metaregression
figure
participant
height
metaregression
quality
assessment
quality
evidence
testex
scale
assess
study
quality
see
table
range
testex
score
commonly
meet
criterion
study
quality
include
group
similar
baseline
titrationprogression
relative
training
intensity
program
statistical
test
’
result
report
commonly
meet
criterion
include
complete
reporting
outcome
datum
include
measure
variance
point
estimate
measure
andor
report
adherence
intervention
potential
bias
review
process
review
unique
feature
inclusion
master’sdoctoral
thesis
include
theses
datum
analyse
great
confidence
finding
review
review
screen
abstract
separate
database
addition
referencecitation
checking
hope
entirety
literature
rom
vast
majority
relevant
literature
include
inclusion
criterion
purposely
keep
simple
lenient
reason
use
testex
scale
provide
gauge
study
quality
say
review
suffer
meaningful
limitation
firstly
inclusion
thesis
result
inclusion
datum
undergo
peer
review
process
rigorous
publish
datum
secondly
effort
manuscript
indicate
subgroup
regression
analysis
deem
exploratory
worth
reiterate
analysis
lack
datum
statistical
power
confident
inference
finally
effort
obtain
datum
possible
unable
obtain
datum
possible
result
review
meaningfully
different
datum
available
discussion
article
aim
review
metaanalyse
effect
rom
rt
range
outcome
major
finding
systematic
review
metaanalysis
rom
rt
appear
modest
impact
outcome
interest
outcome
pool
impact
rom
trivial
small
example
rr
range
show
difference
condition
ci
change
favour
httpsosfioahjnf
result
suggest
different
rom
appropriate
different
goal
example
train
specific
performance
outcome
eg
partial
squat
rm
powerlifting
competition
appear
training
similar
rom
maximise
improvement
trivial
small
margin
result
strongly
suggest
principle
specificity
apply
rom
–
benefit
modest
commonly
assume
look
outcome
group
category
eg
muscle
size
strength
difference
result
prom
largely
trivial
say
noteworthy
effect
size
small
magnitude
directionally
favour
utilise
resistance
training
prove
effective
k                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           sometric
training
different
muscle
length
long
muscle
length
training
result
great
increase
muscle
size
evidence
directly
compare
effect
prom
rt
different
muscle
length
muscle
size
reasonably
consistent
study
exist
area
review
pedrosa
et
al
show
great
quadriceps
growth
follow
prom
rt
long
compare
short
muscle
length
similar
previous
study
mcmahon
et
al
see
similar
result
vastus
lateralis
maeo
et
al
see
great
hypertrophy
biarticular
segment
hamstring
follow
rt
long
muscle
length
compare
rt
short
muscle
length
similar
result
find
sato
et
al
elbow
flexor
study
maeo
et
al
feature
withinsubject
design
compare
l                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         
� ���
�
�                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      �p overhead�$�o �M
elbow
extension
show
great
hypertrophy
head
triceps
brachii
long
muscle
length
condition
finding
noteworthy
long
head
triceps
brachii
train
long
muscle
length
�n #overheadarm�m !neutralarm�M�l �
strategy
important
note
use
rom
necessarily
binary
decision
training
training
prom
analysis
support
hypothesis
perform
prom
rt
long
muscle
length
result
great
muscle
hypertrophy
prom
rt
short
muscle
length
rt
suggest
muscle
hypertrophy
goal
trainee
wish
use
prom
rt
long
muscle
length
training
substantial
support
evidence
concept
resistance
training
long
muscle
length
optimise
hypertrophy
oranchuk
et
al
systematic
review
effect
isometric
training
adaptation
suggest
study
include
isometric
training
different
muscle
length
long
muscle
length
training
result
great
increase
muscle
size
evidence
directly
compare
effect
prom
rt
different
muscle
length
muscle
size
reasonably
consistent
study
exist
area
review
pedrosa
et
al
show
great
quadriceps
growth
follow
prom
rt
long
compare
short
muscle
length
similar
previous
study
mcmahon
et
al
see
similar
result
vastus
lateralis
maeo
et
al
see
great
hypertrophy
biarticular
segment
hamstring
follow
rt
long
muscle
length
compare
rt
short
muscle
length
similar
result
find
sato
et
al
elbow
flexor
study
maeo
et
al
feature
withinsubject
design
compare
    le
length
contribute
great
mtorc
pathway
activation
great
muscle
hypertrophy
prom
rt
short
muscle
length
emerge
evidence
suggest
stretchmediated
hypertrophy
play
substantial
role
human
recent
investigation
warneke
et
al
gastrocnemius
muscle
show
substantial
hypertrophy
stretch
maximally
dorsiflexed
position
hour
day
week
body
literature
convincing
consider
combination
evidence
converge
suggest
training
long
muscle
length
likely
benefit
seek
maximise
muscle
growth
possible
rt
superior
prom
rt
include
long
muscle
length
possibility
prom
rt
long
muscle
length
–
isometric
contraction
long
muscle
length
–
equal
superior
rt
induce
muscle
hypertrophy
area
require
research
give
effective
prom
long
muscle
length
appear
previous
review
muscle
hypertrophy
rom
overestimate
beneficial
impact
muscle
hypertrophy
specifically
metaanalysis
pallarés
et
al
find
large
effect
size
favour
muscle
hypertrophy
notably
lowerlimb
hypertrophy
analyse
study
include
contrast
look
muscle
hypertrophy
outcome
analysis
reveal
trivial
smd
figure
rr
ci
httpsosfiouvd
difference
likely
explain
inclusion
datum
study
include
upperbody
muscle
group
study
publish
analysis
pallarés
et
al
systematic
review
schoenfeld
grgic
conclude
evidence
suggest
rt
superior
prom
rt
lowerlimb
hypertrophy
effect
clear
upper
body
difference
article
result
theirs
likely
stem
inclusion
trial
publish
publication
schoenfeld
grgic
review
article
surmise
response
rom
rt
musclespecific
sub
group
analysis
figure
compare
upper
vs
low
body
outcome
support
idea
research
helpful
test
hypothesis
study
find
great
distal
hypertrophy
define
muscle
length
origin
follow
rt
prom
rt
long
muscle
length
compare
prom
rt
short
muscle
length
similar
proximal
hypertrophy
say
subgroup
analysis
regional
hypertrophy
figure
show
small
smd
ci
–
favour
rt
distal
hypertrophy
trivial
smd
ci
–
favour
rt
proximal
hypertrophy
httpsosfiowdjxg
rrs
difference
regional
hypertrophy
exist
prom
rt
shorter
long
muscle
length
training
datum
require
conclusion
credibility
important
note
outcome
bodyfat
subgroup
moderator
analysis
proximal
vs
distal
hypertrophy
analysis
base
datum
relatively
underpowered
caution
advise
draw
conclusion
reader
adopt
viewpoint
good
befit
researcher
conceptual
consist
rom
relatively
inconsequential
variable
analysis
underpowered
view
range
motion
research
area
infancy
lack
datum
require
come
sort
consensus
topic
second
viewpoint
aim
minimize
q                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            � ����                                                                                                                                                                                                                                                                                                                                                                                                     �p�t �e
error
good
befits
practitioner
range
motion
vs
little
practical
downside
benefit
strategy
small
uncertain
likely
worth
adopt
provide
contraindication
personal
preference
load
availability
injury
management
practitioner
recognize
value
small
effect
existence
relatively
uncertain
small
potential
gain
meaningful
coach
athlete
competitive
recreational
alike
conclusion
outperform
prom
outcome
type
effect
size
range
trivial
small
smds
well
overall
rr
condition
ci
change
favour
appear
small
difference
outcome
depend
exactly
rom
manipulate
eg
short
vs
long
muscle
length
regional
hypertrophy
coachesathlete
wish
adopt
rom
strategy
appropriate
goal
principle
specificity
likely
apply
rom
training
usually
replicate
rom
outcome
interest
approach
good
�s negative�r false�Z�q �9
condition
lateral
medial
head
triceps
brachii
see
great
hypertrophy
contrast
study
study
stasinaki
et
al
find
significant
difference
triceps
brachii
long
head
hypertrophy
follow
prom
rt
long
vs
short
muscle
length
long
muscle
length
generally
appear
result
great
degree
passive
tension
passive
tissue
begin
reach
maximal
length
provide
resistance
increase
muscle
length
tension
suggest
activate
mtorc
pathway
associate
muscle
hypertrophy
great
degree
passive
tension
prom
rt
long
muscle
length
contribute
great
mtorc
pathway
activation
great
muscle
hypertrophy
prom
rt
short
muscle
length
emerge
evidence
suggest
stretchmediated
hypertrophy
play
substantial
role
human
recent
investigation
warneke
et
al
gastrocnemius
muscle
show
substantial
hypertrophy
stretch
maximally
dorsiflexed
position
hour
day
week
body
literature
convincing
consider
combination
evidence
converge
suggest
training
long
muscle
length
likely
benefit
seek
maximise
muscle
growth
possible
rt
superior
prom
rt
include
long
muscle
length
possibility
prom
rt
long
muscle
length
–
isometric
contraction
long
muscle
length
–
equal
superior
rt
induce
muscle
hypertrophy
area
require
research
give
effective
prom
long
muscle
length
appear
previous
review
muscle
hypertrophy
rom
overestimate
beneficial
impact
muscle
hypertrophy
specifically
metaanalysis
pallarés
et
al
find
large
effect
size
favour
muscle
hypertrophy
notably
lowerlimb
hypertrophy
analyse
study
include
contrast
look
muscle
hypertrophy
outcome
analysis
reveal
trivial
smd
figure
rr
ci
httpsosfiouvd
difference
likely
explain
inclusion
datum
study
include
upperbody
muscle
group
study
publish
analysis
pallarés
et
al
systematic
review
schoenfeld
grgic
conclude
evidence
suggest
rt
superior
prom
rt
lowerlimb
hypertrophy
effect
clear
upper
body
difference
article
result
theirs
likely
stem
inclusion
trial
publish
publication
schoenfeld
grgic
review
article
surmise
response
rom
rt
musclespecific
sub
group
analysis
figure
compare
upper
vs
low
body
outcome
support
idea
research
helpful
test
hypothesis
study
find
great
distal
hypertrophy
define
muscle
length
origin
follow
rt
prom
rt
long
muscle
length
compare
prom
rt
short
muscle
length
similar
proximal
hypertrophy
say
subgroup
analysis
regional
hypertrophy
figure
show
small
smd
ci
–
favour
rt
distal
hypertrophy
trivial
smd
ci
–
favour
rt
proximal
hypertrophy
httpsosfiowdjxg
rrs
difference
regional
hypertrophy
exist
prom
rt
shorter
long
muscle
length
training
datum
require
conclusion
credibility
important
note
outcome
bodyfat
subgroup
moderator
analysis
proximal
vs
distal
hypertrophy
analysis
base
datum
relatively
underpowered
caution
advise
draw
conclusion
reader
adopt
viewpoint
good
befit
researcher
conceptual
consist
rom
relatively
inconsequential
variable
analysis
underpowered
view
range
motion
research
area
infancy
lack
datum
require
come
sort
consensus
topic
second
viewpoint
aim
minimize


� � 
�
�
�
|
o
Z
K
9
,

	�	�	�	�	�	�	�	�	�	t	e	X	L	>	5	,	#			���������|Sqg\ND9-#	���������^3�hYNC8.%����������|kXJ;*����������~ncVI<.����������vkcoYM>5'���������ugXKA5,!
���������zrg[RE;2$����������seUMC:-%	����������yi_NA4(D��������~jUF/���������{equipment�
equip	wequation�equate�� erloadlerlifter�erk	eric}erectorere�ercise
erate�er�equivocal�!equivalent
Vequipment�
equip	wequation�equate�enzymeq'environmental�#environment	<environlentire.
enter
entedqent�ensure �	ensu	�enlarge�enjoyment	penhanced?enhance7englishK#engineering�
engage
%energystressmenergy�'energetically�'endurancetype	�enduranceS
endur	�!endomysial�endocrine�endeavor �end�encourage
.encode
�	ence*en�
employ+#empirically	sempirical,empiri	Bemphasizeemphasis �emergeZemLelusive
Eelucidate�elongate�ellite�
elite(elicit,elevationvelevated<elevate.element�elderly|
elbowv!elasticityYelasticXel�#eimdrelated�)eimdassociated�	eimdAei	*egeficialK)efforttraining	c
effortdefficient�!efficiency�efficacy	D#efficacious�effectors0effector'effectiveness �#effectively �effectivel
effectFefTediting�	edit�
edemaBed�%ecologicallyjecologi�'eccentriconly	'eccentrically	-eccentric4
eccen	geat	easy!
earth
�
early �
earli"dystrophy
�9dystrophinglycoprotein
�1dystrophinencoding
�dynamic
dynamduration3duchenne
�	duce�!dualenergy�duz	drug	dropsets�dropset�dropoff�	drop�driver�
drive	driv�drinkingU
drink�	drawT
draft�!downstream	�download�	down	
doubt�%doseresponse�	dose�donationXdonate�dominate$dominant�domainmdoctoralHdo
dnadmd
�dlh�dization	�divide�diverse�divergentditional�dition�disturb�%distributiond!distributek#distinguishqdistinctdistalt!disruption�dispute?-disproportionate�#displeasure�displayB%displacementdismissal=+diseasespecific
`disease
b!discussion%discuss �!discrepant#discrepancyN  ydiscrep`discover
�!discomfort�!disclosure�#discernible	y%disassociateldirectly �direction
~direct�dio	�
dimer�difficult�#differently+differentiation)differentiatedL'differentiateTdifferentp!difference0differ�difQdietary�	diet
idictate�#dichotomizepdiatorAdiameter�#dynamometer�exception]
dozenP#dynamicallyHenableexpectexpand�existence�	existexhibit�!exhaustion�exertion		
exertOexerkine
?+exercisetrained�-exercisespecific!+exerciserelatedb/exerciseregulated
Zexerciser
f+exerciseinduced|)exerciseinduce
}!exercisein�1exerciseassociatedexerciseexercis�	exer�execution �#exclusivelyZexciting�!excitation�#excessively�excessive-exceed�example�examine �#examination
	exam�!exaggerate�
exact�exevolve
�evolutionf
evoke�-evidenceinformedKevidence!eventually�eventual�
event
y
evant�evanston�evaluate�ev�etition�etc�	etal6et �!estimation�estimatekestablish�
estab]#essentially	/essential�essence	�!especiallyescape�esery3
error'erroneous�    e
motion
prom
partial
range
motion
range
motion
declaration
ethic
approval
consent
participate
applicable
consent
publication
applicable
availability
datum
material
analysis
script
datasheet
digitize
picture
figure
analysis
text
output
analysis
grade
table
evidence
systematic
search
result
screen
detail
testex
result
available
supplementary
material
funding
fund
lead
investigator
phd
project
provide
renaissance
periodization
acknowledgement
thank
james
wright
eric
helm
provide
useful
feedback
finished
manuscript
compete
interest
author
declare
compete
interest
content
manuscript
u                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   �� �������xng\PB6/!
�
�
�
�
�
�
�
�
�
�
�
t
g
[
N
B
/
!

	�	�	�	�	�	�	�	�	�	n	`	X	E	9	.			���������yp_SH:1&���������{qi]PC7+#����������|rd[SG>-����������yp`PD:-�����������neVD5*"�����������vndZN>1����������yi\ND:/ ������          �@ space
�? outside�> 'extracellular�= !sarcolemma�< membrane
�; beneath�: 'intracellular�9 ize�8 'compartmental�7 fluid�6 etal�5 skel�4 collagen
�3 compose�2 !connective
�1 sheathe�0 tal�/ skele
�. warrant�- cle�, mus�+ !appreciate	�* detail	�) update�( 'comprehensive
�' consult
�& courage�% !discussion�$ 'physiological
�# nuanced�" !interested	�! reader�  mode� ture� na
� general� overview� brief� cover� later� !bodyweight� distinct� tor� fac� !constraint� pende� de
� unravel� !structural� es� chang� molecular� %characterize� structure�
 com�	 'conceptualize� positive	� tissue	� sition� po
� purpose	� denote� core� #differently	�  trophy� hyper�~ scale	�} scopic�| micro�{ #macroscopic
�z urement�y meas	�x detect�w !measurable�v !ultimately�u mps�t synthesis�s tein�r pro�q %upregulation�p eventual	�o stress�n metabolic�m tific�l scien
�k century�j past	�i physio�h progress�g !tremendous�f #onciliation�e rec	�d idence�c ev�b phy�a hypertro�` lose�_ ning
�^ content�] cell�\ satellite�[ accretion
�Z balance�Y net�X shift	�W includ�V #hyperplasia	�U number�T iste�S preex�R ial�Q ax	�P energy	�O intake
�N protein
�M dietary	�L mainly�K en�J driv
�I process�H adulthood�G longterm�F )reconciliation�E construct�D elucidate
�C devoted�B cific�A spe�@ !consortium�? consensus�> represent�= letic�< ath�; skeletal�: field	�9 expert�8 #competition�7 ing�6 los�5 win�4 !relatively�3 predicate
�2 placing�1 judge�0 #bodybuilder�/ attribute�. desirable�- 'strengthpower�, accompany�+ bulk�* #preexisting�) axial	�( crease�' 'operationally�& il�% evanston�$ %northwestern�# !statistics�" #engineering�! !biomedical�  #departments	� canada� hamilton� mcmaster� #kinesiology� bag
� private� !technology� auckland� 'environmental	� sprinz
� zealand	� sports
� alabama� !birmingham� llc� fitomics� australia� vic� melbourne� victoria� institute	�
 school�	 usa� ny� bronx� cuny� !department� vigotsky� d	� andrew� phillips	�  stuart� helms�~ r�} eric�| haun�{ t�z cody�y jozo�x theiusca�w !population�v athletic�3�u �k
approach
overall
result
suggest
variety
rom
good
effect
injury
management
personal
preference
researcher
interested
see
future
study
compare
adaptation
follow
prom
training
different
muscle
length
compare
example
study
examine
muscle
thickness
adaptation
follow
resistance
training
prom
condition
different
muscle
length
condition
ease
future
analysis
andor
replication
future
research
ensure
datum
openly
available
easy
extract
fail
effort
provide
datum
request
abbreviation
rom
range
motion
prom
partial
range
motion
range
motion
declaration
ethic
approval
consent
participate
applicable
consent
publication
applicable
availability
datum
material
analysis
script
datasheet
digitize
picture
figure
analysis
text
output
analysis
grade
table
evidence
systematic
search
result
screen
detail
testex
result
available
supplementary
material
funding
fund
lead
investigator
phd
project
provide
renaissance
periodization
acknowledgement
thank
james
wright
eric
helm
provide
useful
feedback
finished
manuscript
compete
interest
author
declare
compete
interest
content
manuscript
   ����������ocUE9%����������~ocRB0&�����������{siZPE6(	���������{j]=3&������ycZM?5)
�
�
�
�
�
�
�
�
�
�
x
i
Z
G
<
2
*

		�	�	�	�	�	�	�	�	�		u	j	_	R	:	-	!			 ����������ziPF91'�����������xo_OA3)���������gWK7+��������|obRI3&���������zp_O>3)���������zlaUF4%���������zi]OC5! �D !limitation�C )implementation�B modeling	�A diator�@ indirect	�? jessee�> #participant�= mediation�< )betweensubject	�; causal�: !confounder�9 properly�8 mediator�7 nuzzo	�6 lation�5 corre�4 #shortcoming�3 #inferential	�2 remedy�1 tributory�0 %experimental�/ sociation	�. course�- noise�, relation�+ cor	�* insist�) !covariance�( attenuate�' error�& struc�% 1variancecovariance�$ dominate�# #variability�" !biological
�! meaning�  #correlation� late� calcu� #importantly� !dependence� #statistical� weak
� ational	� associ� cloud� tend� !perimental� ex� +straightforward� gross� typically� '
muscle
size
� #measurement� ic� dynam
� closely� -
training
study
�
 fig�	 !
practice

� trainee
� outpace	� vastly� !discrepant� #appreciably� !multijoint� #requirement� %coordination	�  sidere� #complicated�~ linearly�} ics	�| mechan�{ intrinsic	�z factor�y )noncontractile	�x matter�w !complicate�v 1multicompositional�u !multiscale	�t simple
�s reality�r crual�q ac�p 'associational�o #secondarily�n #theoretical�m argu	�l debate�k hotly
�j remains�i cause�h tory�g contribu�f loenneke�e !contention�d !experiment�c rie�b se�a catalyze�` #uncontested�_ go�^ %particularly�] way	�\ vidual�[ indi	�Z second�Y —�X necessary	�W reduce�V argument�U #independent�T gain�S ed
�R proceed�Q lack�P 3strengthhypertrophy�O #presumptive
�N buckner�M hold�L gue�K ar�J duce�I +forcegenerating�H #fundamental�G additive�F forc�E tenet�D basic�C icate�B pred	�A notion	�@ equate
�? premise�> 1hypertrophyoriente
�= clearly�< ature�; liter�: mind�9 decipher�8 help�7 !physiology
�6 ongoing�5 exciting	�4 sponse�3 sign	�2 widely�1 %standardized�0 'compositional�/ +ultrastructural	�. nature�- hu�, bril�+ prime�* 'energetically�) spatially�( pertrophy�' hy�& ological�% physi
�$ roberts�# rationale�" consider
�! prudent�  !opposition� high� !phenomenon� gest� sug� transient
� account	� ponent� word� relative
� coplasm� sar� -disproportionate� A
sarcoplasmic
hypertrophy
� 'comparatively� !eventually� manifest� #
daugh
ter
� split� C
myofibril
expansion
cycle
� !compelling� jorgensen
�
 chronic�	 line� A
conventional
hypertrophy

� genesis� #myofibrillo� splitting	� series� associate� rize� theo�  %myofibrillar
� regular�~ !weeksmonth�} 'intramuscular�| stitutes
�{ average�z %triglyceride�y glycogen�x substrate�w store�v tain�u iological�t cupy�s oc�r ion	�q enzyme�p ribosome�o brils�n myofi�m ment
�l environ�k estimate�j nent�i compo�h major
�g release�f age�e stor
�d calcium�c site�b %sarcoplasmic�a !production�` #recruitment	�_ neural�^ sarcomere
�] contain�\ unit�[ tractile�Z con�Y reticulum
�X plasmic�W sarco�V call�U organelle�T #specialized�S ulum�R retic�Q 'mitochondrial�P )multinucleated	�O bundle�N !fascicular�M separate	�L occupy�K #vasculature
�J consist�I myofibril�H nelle�G orga
�F suspend	�E medium
�D aqueous�C !sarcoplasm�B cellular�A intra
`" SH<,
��������sfWK>.#	
�
�
�
�
�
�
�
�
�
y
b
Q
>
/
&


	�	�	�	�	�	�	�	�	�	�	}	t	f	X	H	>	1	'		������������}qdYPD;0����������wVC5$
�����������vjaT@2)�����������zoe\RG8*����������rcYNB4%$����������zpcUJ<,
�����������~�kaUK>0"������~sk`TK@7-"��������|tj\RH6+����������vd[PB;1ap�
evoke�generateYgenerally=general	gene@gender�	gata*'gastrocnemius�
gapdhagap�ganize
	gain�!gadolinium�g�
futureNfutileifusionW	fuse�#furthermore�funding�#fundamental�#functioning(!functional*function&
fullyOfulltextFfullrom�	full�fu
/fruitfuly	frey�!frequentlyfrequentGfrequency7frequenb#freeweights�!freeweight�freedom�	free�fre:framework=!fractional
�founding�foster�
forth	fortiformative�formance	�formUforecast	�-forcetransducing
�)forceproducing�forcepro�+forcegenerating�forced�
forcef	forc�football�	foot�fonseca�followup�following�
followJ	fold�fol�	focus
focal
Wfo
)flywheel	Mfly�	fluxc
fluid7#fluctuationzfluctuate�	fluc�	flow�	flnco
flict
flexor
�flexion�flexibleq
flann�fixfitomics�)fitnessrelated
2fitness�fitVfisher�	firm
-firing�finland�	fink	finite	Zfinger?#finegrained�finding �	find �financial�finally
final	5filamincbagdependent
L#filamincbag
�filamincn'filaminbagyap� Cfilaminbagmtorcyapautophagy�!filaminbagyfilaminrfilament�)figuredownload�figureXfig
fieldeg�
field�ficultYfibrous�!fibroblast
�fibertypey+fiberassociated�
fiberOfew	�ferentlfer�
femur	Vfemoris$female		feel�	feed
�fee
8fedstate
fects
	fectUfeasible�favour�
favor1fatigued9fatiguebfatb!fasttwitchz#fastedstate	fast �fashion�	fash�!fascicularNfascicle�farthing	!farefamily=!familiarlyL'falsepositive�
falserfak
�fairly�failureh	fail�faculty�factor�	fact!facilitateJfacfa	1eye�extremity�'extrapolation�#extrapolate �/extramyofibrillar�extract�'extracellular>
extra�external!exterior
�extentextensor�#extensively�extensive�extensionBextended2extend(
exten�!expression
xexpressNexpose�explosive	)explore �#exploratory�#exploration!explicitly �#explanation	�explainbexpert�%experimenterf)experimentally�+experimentalistZ%experimental0!experiment�!experience8experi�
exper~expense�   �expend7expectexpand�existence�	existexhibit�!exhaustion�exertion		
exertOexerkine
?+exercis'forcevelocity�gastroc�
fifthrgracilis;fourth	half
]h guideline8
guidegue�%growthrelate'growthoriente�
growthW	grow�+groupsconditionM	groupa
gross	grip �
grgic'
great gravity
�'gravitationale
grant
4	gramUgradient
grade�%gprdependentM	goto�	good �goldberg
�	goal �go�!glycolytic`!glycolysis3glycogenygluteus�glutealglutamineI-glucosephosphate�glucose�globulingglobal�gli<gle�
gland�	give �giv]
girth	gin	;gillieL
giantQgh;
geste�	gest�georgeHgenetic�genesis�genericq!generation�)fatigueability!generating)#freeradical�   ��������|n`SF:,$���������yh]N;)����������}rg_SE=1$���������rdWL?1& ����}n_VK@5(
�
�
�
�
�
�
�
}
s
g
]
Q
H
=
-


 	�	�	�	�	�	�	�	�	q	c	R	G	=	3	)			��������rf\Q@. 
���������tjVKA8-����������wl\QE5$����������tiYH91)���������ukXJA3'���������vi]I;-����������}q`UB2'            �H dence�G confi�F !strengthen�E 'triangulation�D sight�C #observation	�B signal
�A summary�@ gene	�? finger�> zinc	�= family�< gli
�; variant	�: intron�9 %unidentified
�8 theless
�7 sociate�6 snp�5 %polymorphism�4 cleotide�3 singlenu�2 underpin�1 )transcriptomic	�0 naïve
�/ recruit�. hallmark�- %nonresponder�, responder�+ socalled�* ence�) quire�( #functioning�' multiple�& #orchestrate
�% capture
�$ acutely�# mal	�" riboso	�! innate�  resident� als� individu� 'substantially� fact� !regulation� polygenic	� expect	� system� tegrates
� complex� integrate� !headtohead� tributing
� refined	� repair	� damage� speculate� th� rd� correlate� #daystoweeks�
 !mediumterm�	 water� deuterate� fedstate� #fastedstate� %postexercise� !surprising	� extent	� ercise	� postex�  h� instance�~ scenario�} #nonetheless�| er�{ #comparative�z !unilateral	�y respon�x trial�w !subsequent�v drink�u soy�t %isoenergetic�s )proteinmatched�r anabolic�q milk�p skim	�o bovine�n turnover�m ple�l exam
�k broadly�j !phenotypic�i align�h !longerterm�g evant�f rel�e able�d avail�c )siondetermined�b infu�a hours�` shortterm�_ 'animalderived�^ ity�] qual�\ )bodyweightmeal�[ proteinkg�Z g�Y maximally
�X permeal�W !estimation�V make	�U margin�T isolated�S %translatable�R #mealinduced�Q mixed�P dose	�O ingest�N %doseresponse�M -hyperinsulinemia�L resultant�K rise
�J trigger	�I robust�H 'aminoacidemia�G 'sensitization�F %carbohydrate
�E leucine�D ami�C essential�B lift�A cise�@ exer�? pande�> #methodology�= infusion�< extensive
�; isotope	�: stable�9 /infusioningestion�8 undertake�7 #challenging�6 -methodologically�5 !admittedly�4 hour�3 !exercisein�2 meal�1 'understanding�0 !persistent�/ itive�. pos	�- postrt�, come	�+ exceed�* rate�) /ingestioninfusion�( 1hyperaminoacidemia�' #synergistic�& ingestion�% acid�$ amino�# !responsive�" %surprisingly�! fold�  fluctuate	� person
� healthy� young� locus� accre� mpb� breakdown� reconcileM� �
increase
individual
muscle
size
contribute
increase
individual
strength
� ultimate	� answer� formative� unin� !odological� meth	� nuance� swell� founding
� opinion� claim
� element�
 forcepro�	 credulity	� strain� eye� )methodological	� hinder� ent� !impossible� jury� %conservative�  -strengthhypertro
� imental�~ exper�} tight
�| quality	�{ ration�z du�y fruitful	�x hardly�w mm�v elbow�u quent�t subse
�s augment�r rest�q randomize�p tive
�o perspec�n question	�m proach	�l clever�k !obtainable�j un	�i futile	�h assign�g randomly�f %experimenter�e 'inconceivable�d dependent�c ality�b #problematic�a arguably�` !reasonable�_ %contributory�^ lish�] estab�\ tion
�[ correla�Z +experimentalist�Y !suboptimal�X lecte�W col	�V nearly�U fect�T ef�S unbiased	�R obtain
�Q collect
�P typical�O confound�N residual	�M assume�L )datagenerating
�K ideally�J %tweensubject�I !adequately�H wish�G tionship�F rela�E 'withinsubject   ����������sj\C6%�����������~sf\N@4(����������yjbPE<,����������vl[N?3&��������}r\LA0!
�
�
�
�
�
�
�
�
�
�

v
c
V
D
5
&

	�	�	�	�	�	�	�	�	�	u	e	U	<	)	 		���������}maTE9.%����������wk`RD5*�����������wm_OC1$��������{ri[PC9(����������mbNA4%���������yk\RB9-��������}ncZG3!
�N routine�M %volumematche�L )moderatevolume�K 'volumeequated�J ale�I sound�H seemingly�G frequent
�F viduals�E untrained�D timeframe
�C persist
�B display�A /resistancetrained
�@ truncat�? #postworkout�> baseline	�= return�< elevated	�; quency�: fre�9 !tification�8 quan�7 frequency�6 readjust�5 #incremental�4 /supercompensation�3 ery�2 recov	�1 active�0 tolerable�/ overreach�. culminate�- #programming�, empirical�+ constant�* ope�) welldevel
�( veloped
�' underde�& )specialization�% ingly�$ )underdeveloped�# vance�" rele�! !particular
�  trophic� scription
� minimum� 'approximately� #exploration� igate� mit� ceivably� 'progressively� #periodizing� pose
� journal� guide� starting� ite� lim� 'incrementally	� weekly	� ualize
� individ� #ancetrained	� resist�
 !hypertroph�	 )hypothetically� lifestyle� #undoubtedly� !overtraine
� plateau� %increasingly	� confer� !hypothesis� hormesis�  cept� curve
�~ ushaped	�} invert�| fer�{ bin�z %unresponsive
�y largely�x tocol�w %sorptiometry�v ab�u xray�t !dualenergy�s sensitive�r !ultrasound�q image�p resonance�o magnetic�n stratify�m grade	�l higher�k cer�j tween	�i driver�h identify	�g ellite�f sit�e #tracellular�d +volumedependent�c quantify	�b metric	�a viable�` kg�_ c�^ b�] ber�\ num�[ speak	�Z devote�Y implement
�X session	�W repeti�V !accomplish
�U vantage�T ad�S atetohigh�R perceive	�Q rating�P #displeasure�O !discomfort�N lowloads�M #furthermore�L tem�K sys�J 'neuromuscular�I tax�H 'timeefficient�G 3hypertrophyoriented�F !prioritize�E !comparable
�D sidered
�C exercis	�B single�A !freeweight�@ modality
�? genetic�> !submaximal�= +interindividual�< #substantial�; %respectively�: ≥�9 quadricep�8 threshold�7 terminate�6 %betweengroup
�5 tensity�4 'inconsistency�3 lar�2 simi�1 mix�0 ditional�/ tra�. #traditional�- mild�, %preferential�+ bfr�* flow�) ii�( %typespecific�' posit�& 'periodization�% inference�$ #preliminary�# prior�" !potentiate�! -strengthoriented�  block� cycle� /hypertrophyorient� initiate
� amplify� %intrasession� intraweek� anabolism� %amalgamation� ably
� conceiv� 'contradictory
� pathway	� kinase� selective
� combina� #possibility� fail� -proofofprinciple
� insight� erate� mod	�
 binary�	 gap� variance	� sample� neity� heteroge� colleague� lopez
� network� !correspond�  !conditions� eat�~ moder�} %moderateload�| ×�{ lightload�z ≤�y highload�x pare�w +loadindependent	�v accord�u irrespec�t true�s #subanalysis�r −�q interval�p !confidence�o 'corresponding�n pool	�m tively	�l respec�k spectrum�j somewhat�i fort
�h lowload�g match�f advantage	�e ically	�d specif�c level
�b explain�a ancie
�` discrep	�_ scheme�^ light�] giv�\ zone�[ challenge	�Z emerge�Y quest�X #competitive
�W maximal�V 3
hypertro
phy
zone
�U presence�T mvc�S voluntary�R imum�Q max�P rm�O !percentage
�N express�M ploye�L em�K -evidenceinformed
�J believe�I !refutation
� � pbTH9.$����������xj�aVL?/"���������}qdP=1(
�
�
�
�
�
�
�
�
�
�
{
p
f
[
P
4
"


	�	�	�	�	�	�	�	�	�	�	y	a	R	<	+		������{gP8	�����n^M9,�����������o[MF;2'�����������~t�gVB6+��������|l0\OA0&�����������saN=*���������xh[M=1"���������|pdRH:#���������xhYMAividuindivid!indigradientinertial	%inducible�induce�!individual �individuindivid!indirectlyindirect@indicatorf!indicative�indicate �	indi�#independent�	inde	�'incrementally#incremental5%increasinglyincreasedVincrease'incorporation�#incorporateW'inconsistency�%inconclusive
'inconceivablee%incompletely
Qinclusion �include �includ�incline�incidence
cinception�inbetween
gination]!inadequate�inability�	imumR#improvement�improved	�improve�!impressive�!impossible�impose?#importantlyimportant	!importance	imply?#implication�implicate&)implementationCimplement�impinge�impaired�impair0
impactDimmune)immobilization
�#immediately�imental	imal	U
image�!illustrate
illness	�ilk
�ilarly�il�iiaNii�	igfsO!igfinduced
�igfieaV
igfecK
igfebJ
igfeaIigfbpsG
igfbpHigf
u
igateie^identify�)identification2%identifiable	bidence�	iden	zideallyK	idea�ics�icehockey	'
icate�icallye	ical~ic'ibandspanningVial�)hypoxicinduced�hypoxic�hypoxia#)hypothetically	#hypothesize �!hypothesis!Ehypothalamicpituitarygonadala#hypoosmoticH7hypertrophytriggering
�9hypertrophystimulating
D3hypertrophyspecific�1hypertrophysensing�-hypertrophysense�3hypertrophyoriented�1hypertrophyoriente�/hypertrophyorient�)hypertrophyorip5hypertrophymediating1/hypertrophyinduce
�7hypertrophyassociated�#hypertrophy'hypertrophies�%hypertrophic �!hypertroph
hypertro�#hypertophic�#hyperplasia�-hyperinsulinemia�hyperemia�1hyperaminoacidemia�
hyper�hyldahl�!hydrolyzed,hydrolyze8!hydrolysis5hydrogen�hydration�hydrated�hy�hw�
human�
hubal�hubthu�%httpsosfioje?9httpsgtexportalorghome�
howev�
hours�	hour�
hotly�	host
�horwarth	&hormoneshormonal
hormo�hormesis	hope	�homodimeru#homeostasis�	hold�%historically
�!historical	�
hippo
Nhip�hinder�'highthresholdn)highrepetition
 highrep
highly�highloadyhighlight �'highintensity�higher�	high�higbie	hickson	�'heterogeneity �heteroge�
heter	�hermetic�hepato1hensive
!henceforthnhelpful�	help�
helmshek_heighten�height�
heavy(heavily �#heavierload�
heavi�healthy�health�!headtohead	head�haustion		haun|hardlyxhappen
hap
	hand�
hance�hamstring�hamper�hamilton�hallmark.#halfmaximal[   >halflive�	half
]h guideline8
guidegu!heightened}imbalanceU#inefficientRinactive7#inclination4!highvolume$%highervolume#impracticalinjury �injure�injection�!initiation�initiate�initial�!inhibitory4inhibitor!inhibition$inhibit	�inherent 
ingly%/ingestioninfusion�ingestion�ingest�ing�/infusioningestion�infusion�	infu�%infrequentlyeinform�influx�influence �
influ�%inflammatory�inferior\#inferential3inference�
infer �infection�!infeasible	�
E 6+����������~paQC+���������yjXC2 ��������seSH5
�
�
��
�
�
�
�
�
~
r
b
Q
E
8
'


	�	�	�	�	�	�	�	�	~	q	c	T	A	3	&			�������h������}tiVK</%�����������sfXJ9,"	 ����������xk\L@7-�����������{pbQB5)������������vg]MA6(
������w�����|qh^TH3"����������{iWE-������|rg^TJA6'������cle   %interstitial1longermusclelength�%longermuscle�%longerlength�%longduration�long,lohmann1logicallylogicalB
logic:loenneke�
locus�locate*locallyUlocalizeq
local�%loadspectrum	�loading/#loadinduced�+loadindependentwloaded/	load)	lnrrZllc�
liverE	livelittle
�!literaturew
liter�	list�	lish^	link
�linearly�linear�	line�limitedG!limitationD
limit�	limb6limlikewise	�
liken�likely!likelihood/	like�lightload{#lighterload�
light^lifter�	lift�lifestyle	life
�lie�liberallyleydig_
levelclev�leukemia3leucine�
letic�	less	lenientRlengthy�lengthten�#lengthening�lengthen�
length-	lend�lehman�legAlection�
lecteX
learn �	lean	lead �lbmlation6laterally
�lateralisMlateral�
later#latencoding	latelat�lastlyK!laseviciusplarginineWlargely�
largejlar�lamina�)lactaterelatedKlactate	lack�!laboratoryG
label
�lJkubota�knowledgeknowYknockout�
knock�	knee�kinetic�#kinesiology�kinematic"kinaseξ-kinaseskinase�kg�keywordTkeyj
keogh	�kelley�	keep�keeler kda�kassianox#jyväskylä�	jury�junction�	jump�
judge�ju�	jozoyjournaljorgensen�
jones	4'jointspecific�
jointhjnk"jn}jjh�jessee?	jeff�	jari�
james�j�ize9
iusca_ity�
itive�	itga6	item9iteistics	�	iste�
issue
>isotope�isotonicK'isometricallyisometricisolated�isolate!isokinetic	#isoinertial	,isoform5%isoenergetic�iso	$ischemic�ischemia�%irrespectiveVirrespecuionriologicaluinward�#involvementinvolve �involv#involuntary�#investigateOinvert�!invariablyintuitive
intron:%introductionXintrinsic�%intriguingly�intricate�intraweek�%intrasession�'intramuscular}/intrafascicularly�'intracellular:
intraA%interventionLintervalq%interstitiumD#intersperse
!intersetm)interpretation�interpretinterplayZ'international@)intermittently%intermittent"#intermedius	0%intermediate�#interleukin2+interindividual�%interference	�interfere�'interestinglyr!interested"interest�#interaction�interact�
inter	Xintensitycintensi	�intense~intend�integrity�)integrinlinked
�1integrinassociated�integrin
�!integrated-integrateintegral-intake�#insulinlikeuinsulin5#instruction�'institutional�institute�instead�instance�insist*insight�inserie�inorganic�
inokikinnervate�   #innate!injury �injureizquierdozlinnamovinsertion@#inscription<leverage*lumbar�
ltypeYlowvolume'lowrepetition
lowloads�lowloadh%lowintensityElowerlimb+lowerbody�
lower�low5	loss}	lose�los�
lopez�	look4longterm�
longo�-longmusclelength�)longitudinally
�%longitudinallongevity
\!longerterm�!intriguing
   � ���������wk\SJ=/$���������xl^N@,!��������~tj\I?4%����������vj_ULC:-����������zqh[RC6#
�
�
�
�
�
�
�
�
�
x
m
b
S
H
>
3
'


	�	�	�	�	�	�	�	�	�	l	Y	I	8	%			����������sgZQD0%����������sfYF8/
�����������j\NB7-�����������zj_TE<2'���������}�6+"���������rbXND8#       	�H priori�G ative�F neg�E +sistancetrained	�D status�C meet�B mize�A maxi�@ !obligatory
�? dispute
�> versial	�= contro�< ure�; !prediction�: accuracy�9 %underpredict	�8 people	�7 expend�6 validate�5 )selfdetermined	�4 predic�3 devel�2 rir�1 anoth�E�0 �
set
endpoint
trainee
complete
final
repetition
international
journal
strength
conditioning
phillips
s
m
steele
j
vigotsky
d
possible
repetition
attempt
ed
definitely
achieve
momentary
failure
{�/ �{
trainee
reach
point
despite
attempt
complete
concentric
por
tion
current
repetition
deviation
prescribe
form
exercise
	�. entire�- !integrated�, cohesive�+ !collection�* ply�) !anatomical�( attention�' #ternatively�& reinforce�% regularly�$ velopment�# ming�" earli�! easy�  mid� poststudy� trunk� delay� !chillibeck� novel� recur� liberally� plex� skill	� involv� sense� logically� fix� slightly	� select� 'autoregulated� rauch� %undetermined� atic� wheth	� choose�
 computer�	 rotation� -sessiontosession� bazvalle	� rotate� !frequently� ty� varie� crude� !cumference�  cir	� midway�~ #freeweights	�} switch�| nhout�{ aere�z !schwanbeck�y tic�x synergis�w 'complementary
�v agonist
�u respond�t !stabilizer�s synergist
�r expense�q howev	�p afford
�o freedom�n selec�m lengthten�l seat�k !demonstrat�j stretched�i sion�h exten�g hand�f vasti�e )preferentially
�d tension�c lie
�b brandao	�a foster
�` lection�_ nonvaried�^ extremity�] perience	�\ varied�[ costa	�Z ilarly�Y sim�X %volumeequate
�W uniform�V lunge�U smith
�T fonseca�S ample�R 'architectural�Q #interaction�P !nonuniform�O 'hypertrophies�N /intrafascicularly
�M inserie	�L neuron�K motor�J innervate�I #subdivision�H -threedimensional�G carry
�F diverse�E tachment�D establish�C pull�B plane	�A pulley�@ cable�? free�> volve�= selection�< cular�; influ�: %machinebased�9 sec�8 rule�7 #tematically�6 !efficiency�5 %preservation�4 fash�3 %consistently	�2 tinely�1 rou�0 erlifter�/ pow�. adapt�- 'consideration
�, workout�+ buffering�* gle
�) phillip�( sin�' fly�& chest	�% tition�$ repe
�# dropoff�" senna�! #singlejoint�  multi� impaired	� dality� mo� #conceivable� adult� !moderation� !volumeload
� tralize� neu� ute� min� pair� longo	� cising� !peripheral� bly� conceiva� !centration� multiset	� minute� blunt�
 rps�	 psk� mckendry� !overshadow	� modest
� tuation� fluc� relevance� doubt� cast�  'concentration� hormonal�~ ical�} crit�| +exerciseinduced�{ regulate�z #fluctuation�y systemic�x ulate�w spec�v elevation�u #insulinlike�t %testosterone
�s hormone
�r prevail�q ented�p )hypertrophyori�o organiza�n !henceforth�m interset	�l ferent�k !distribute�j !persession�i cap�h !recommenda	�g spread�f permuscle�e %infrequently�d %distribution�c cy
�b frequen�a scrutiny�` #
waste
set
�_ iusca�^ plateaus
�] ination�\ ume�[ vol�Z interplay	�Y manage�X !standalone�W day�V %irrespective�U %metaanalytic�T !tweengroup�S ly�R #nonexercise�Q dif�P oxide�O deuterium
: �.k��������~rc�SG:*����`z������9�oUq�BJ�.�j����!��h�2�|qg\A4 �
�
�
�
�
�
�
�
�
�
�
{
m
X
N
=
2
&�


�
	�	�	�	�	�	�	�	�	��	�	y	l	a	S	C	5�	$		�{������������q`PG<,O`p���,�������@�vj[K<,��������tc�V[I2#�������������tk\OD5&����~ui[N=0#	��������H7z(������!� 	pcr +p���~sg`   )postresistanceN'phosphorylate:placeboX	precursor�#powerlifter)ph!pkc�+phosphocreatinepossess6'postoperativepremature>!perpetuate�%perturbation�!presumably�plasma�pretraine�pituitary�pervasive�
piezo�
prado��#phosphatase�-phosphoproteomic�!powerpoint�	ppxy�prolongr!#proliferate�'progressivelyprogress�#programming-programMprofile%professional �!productionaproduct�produce�process�proceed�#problematicbproblem�probably
Yproachmpro�private�prisma�!prioritize�prioriH
prior�	plus!prevailing�phosphate�pressure�postulate�#polypeptide#postmitotic�principle�principal

prime�primary �primarilyapri	�!previously �previous �prevent �prevailrpretopostpretest	�#presumptive�
press�%preservation�present�presenceU%prescription �prescribe �/prepostexhaustion�prepost�premise�#preliminary�preferred7)preferentially�%preferential�!preferences#preexisting�'preexhaustion�preexhaus�#preexercise�
preex�'predominantly
#predisposed�!predisposevpredictor�!prediction;predicate�predic4%predetermine	�	pred�precludeRprecede	!precaution|pre�pragmatic	L	prag	u%practitionerpractice �practicalWpractic�%powerlifting�!powerlifte	�
power �pow�!potentiate�#potentially �potential�#postworkout?posturepoststudypostrt�)postexhaustion	j%postexercisepostex'posteccentric
�	post�possibly
possible3#possibility�positive#positioning:position �
posit�posedly�	posepos�!populationw!popularity�poorly �	poor�	poolnponent�%polymorphism5polygenic
point�podpopn
�ply*ployed�
ployeM#plification	�	plex+plethysmographyplethora\ple�pld'
plcγplayer	(	play^plausiblyplausible%plausibility	�plateaus^plateau
plate	IplasmicXplantaris
�)plantarflexion�plantar
�7planningperiodization	�
plane�	plan	�
plain	�placing�placement�
placepitchingpipphysiquev!physiology�'physiological$physio�physicalV
physi�phy�+phosphorylation
�)phosphorylated
�'phospholipase
�5phosphatidylinositol%phosphatidic
�phillips�phillip�!phenotypic�!phenomenon�!phasically
	phase6pertrophy�pertain �#perspective	�perspecopersonal�person�!persistent�persistCperset�!persessionjpermusclefpermitpermeal�!peripheral�#periodizing!periodized
periodize
'periodization�period�
perio	�!perimentalperience�	peri	�#performance�perform`� 1perforcpereira+!perceptual	o!perc!performingw%postadaptives!precedenceTposturalMproperty�properly9
proper-proofofprinciple�pronounce�promoterpromotepromise[	prom�prolongedn#prolinerich�'proliferation�prepared�!physically�/pharmacologically�� partiallyppotento3phosphofructokinasee'phosphorylasedpocket]   passivelyZportionW  * ����������ykaSE5)���������thWM;0$���������~paUE;/"
����������xm\L3'����������thXG;/
�
�
�
�
�
�
�
�
�
z
l
]
O
B
7
'

	�	�	�	�	�	�	�	�	�	�	m	Z	C	;	/	$			����������ve\L@7, ����������shYD7)����������rbS</!���������xk`RF;1%�����������wiZN?7-	�����������ujaPD:/ 	 ���������vlcUG9*            �S norrbrand�R centuate�Q rewrappe�P continue�O put�N cord�M flywheel�L pragmatic�K !tervention�J #welltrained�I plate
�H removal�G releaser	�F custom	�E walker�D efficacy�C cal	�B empiri�A spite�@ ufacturer�? stack�> tilt	�= xforce�< #environment�; gin�: reima�9 pedal
�8 centric�7 user�6 omni�5 nautilus�4 jones	�3 arthur�2 vor�1 fa�0 #intermedius�/ #essentially�. !overloaded�- 'eccentrically�, #isoinertial�+ ther�* ei�) explosive	�( player�' icehockey�& horwarth�% inertial�$ iso�# ceps�" bi�! farthing�  triconly	� concen� crosssec	� higbie	� device� !isokinetic� 'eccentriconly� )concentriconly	� tricep� band� super	� female� merrigan� final
� precede� haustion� °� #nontraining� )recreationally� trindade� !assessment� xweek	�
 tinued�	 exertion� swelling� down� fink� mri� daysweek
� descend� /withinparticipant� tradition�  !successive
� initial�~ n	�} tional�| tradi	�{ divide�z fol�y goto�x followup�w !supposedly
�v posedly�u sup�t #
weak
link
�s build�r !exhaustion
�q prepost�p equation
�o testing�n +bodycomposition�m preexhaus�l cally
�k ecologi�j !popularity�i apparent	�h dition�g nificant�f sig�e dropsets�d pause	�c setend�b lower�a assistant�` #heavierload�_ male�^ #lighterload�] heavi�\ ere	�[ consid�Z !tistically�Y sta�X #preexercise	�W ciable�V appre	�U forced�T post�S #immediately�R decrement
�Q etition�P rep�O ahtiainen�N !indicative�M aptation�L temic�K drop�J nal�I hormo	�H perset�G su�F /agonistantagonist�E 'preexhaustion�D )agonistagonist�C nique�B tech	�A experi�@ geste�? !respondent
�> practic	�= survey�< advocate�; #anecdotally�: superset�9 /prepostexhaustion
�8 dropset�7 !exaggerate�6 hance
�5 cialize�4 overload�3 following�2 advanced�1 el�0 lev�/ tapering	�. option	�- covery�, sparingly�+ %recuperation�* #consequence�) ratio�( +stimulusfatigue�' cises
�& confine�% )conservatively	�$ ployed	�# highly�" #speculative�! !creasingly	�  lifter	� novice� difficult� ationally� oper� poor� #involuntary	� demand� overcome� inability� feel� !volitional� break	� nition� defi	� sensus� 3
momentary
failure
� !continuous� #finegrained� begin� velocity� step�
 )thresholdbased	�	 linear	� purely� exact
� fashion� detrimen� bout� %recuperative	� sitate� neces
�  advance
� decline	�~ pacity�} loss�| !precaution	�{ stepup�z uncertain�y validity�x tolerate�w #machinebase�v !predispose
�u manding�t compound	�s decide�r ation	�q quired�p !lasevicius�o scant�n 'highthreshold�m cruit	�l erload�k valid�j %ecologically�i seek�h #selectively	�g choice�f indicator�e reliable�d mance	�c perfor
�b sustain�a spect�` %musclebuilde�_ %overtraining	�^ marker�] #continually�\ turn
�[ promise�Z #exclusively	�Y ficult�X !comparison�W activity�V physical�U gram�T utilized�S endurance�R bic�Q aero�P rent	�O concur�N #discrepancy�M !nonfailure	�L vieira
�K eficial�J ben	�I negate   �� ���������th_SF4 ���������zk[=,�����������vmaWG6$���������{j]N;1$���������zm^RG=0"
�
�
�
�
�
�
�
�
�
~
u
h
Z
O
C
8
)
 

	�	�	�	�	�	�	�	�	�	|	l	`	E	7	-			���������{tlcUE8/"������������������vfXE8%����������zn]QI=-���������}n^OD2"�����          �0 condensed�/ fu�. encourage�- firm�, troll�+ mak�* cuse�) fo�( !ameliorate�' %organization�& daily�% alternate�$ !mesocycles�# odization	�" deload�! #intersperse�  )highrepetition
� hensive	� compre	� odized� !illustrate� strictly� possibly� periodize� !weektoweek� antretter� !terspersed	� santos� do	� narrow� #examination	� happen� hap� minority� flict� !periodized� fects� undulate�
 
linear
�	 ate	� deline� !phasically	� ganize� principal� %inconclusive� 'lowrepetition
� highrep� 'predominantly�  relevant� !powerlifte
�~ winwood�} keogh�| %loadspectrum
�{ pendent�z inde	�y marily�x pri	�w season�v overtrain
�u accrual�t likewise	�s tation�r adap}�q �
peri
odization
plan
manipulation
training
variable
order
maximize
training
adaptation
prevent
onset
overtraine
syndrome
	�p buford�o dization�n perio�m forecast�l %predetermine�k improved�j !successful�i vari�h 'nonperiodized�g plan
�f onymous�e syn
�d illness�c !occurrence�b formance�a few�` τ�_ q�^ !assumption�] csapo?�\ �
hypertro
phy
training
periodize
goal
maximize
muscle
mass
�3�[ �k
variable
betweenstudy
variance
explain
methodological
factor
give
finding
likely
variation
es
study
likely
attribute
sampling
variation
potentially
individual
participant
level
char
acteristic
study
level
charac
teristic
say
comparison
con
dition
variation
treatment
effect
reveal
little
difference
concurrent
single
modality
training
log
variability
ratio
ci
manipulate
hypertrophy
mesocycle
maximize
increase
contractile
tissue
3�Z m
transference
hypertrophy
q
p
τ
strength
power
�Y arise�X !macrocycle�W mesocycle�V hope�U !ostensibly�T %successively�S %subsequently�R peri�Q organize�P 7planningperiodization	�O cardio�N !infeasible�M apart�L schedule	�K source�J )characteristic�I detriment	�H overly�G #perspective	�F conclu	�E absent�D ogeneity�C heter�B ticipant�A par�@ treatment�? condi	�> random�= tifie�< schumann
�; thereof�: ual�9 !historical�8 plain�7 #nutritional�6 sex	�5 istics�4 character�3 alongside
�2 culprit
�1 intensi
�0 overlap�/ #plification
�. oversim�- adaptive
�, running�+ ance�* endur	�) adjust�( deviation
�' pretest�& delta	�% nation�$ combi�# !unweighted	�" wilson�! tenuated
�  respect� #corroborate� %plausibility� !downstream� rapamycin� mammalian
� inhibit� path� 'monophosphate� adenosine
� essence� #explanation� offer� 'endurancetype	� volved� %interference
� concern
� hickson
� classic	� decade� %colloquially� dio�
 car�	 -aerobicendurance� %concurrently� #participate� !concurrent� ensu	� career� val� cient� timeeffi
�  novelty� less	�~ attain	�} vanced�| od	�{ tified�z iden�y #discernible�x !occasional�w equip�v matic�u prag�t necessity�s #empirically�r =overreachingovertraining�q !stagnation�p enjoyment�o !perceptual�n !subjective�m argua�l wholebody	�k dearth�j )postexhaustion	�i todate�h tric�g eccen�f !regularity�e sider�d !bloodbased�c )efforttraining�b %identifiable
�a notable	�` nology�_ ued	�^ contin�] !noteworthy�\ )nonsignificant�[ timescale	�Z finite
�Y vention�X inter�W technical�V femur�U imal�T #selectorize
f	!�����ufSRA,	3N&
�������������tg�^hWH8.!	��������� �e��������=�����ynf^SE�;/!%������� ����qfX�D7&��H�1�2��������0u/@�gYYK9,���t������A��yi_RD5'��>����������!����~ocR�E5(�
�
�
�R
��
�
�
�
�
�
z
j
[
M�
=
3
"

u��	�	�	�	��	��	�	�k<	�	�	�	�	z	p	f	^	Q	EN	9	/	'�	qk�\v		����$�������rKcZfYN!repetitive"reduced&!remodeling�
prove�region�rabbit�rotation	�#responsible�!protective�reflect�'reinforcement�replicate�pulsatile�%repartitione�#replicationrrapidlyk!regenerate]regimen:#proteolysis7
rapid#
redox'proteinkinase� prou�!regulatory proutineNretain�reside�#replacement�%recreational�quantity�pursue�res�revise�quo�proteomic�� row9realize�redundant~related{robustlyvroutinely�	rhebb#resynthesis7%resynthesize-refute*proposal&regimerosreactive
quiescent�repeat�remove�� prtinduced)rtrps�recipient�#regenerated�%regeneration�-resistancetraine�putative�
rugbyy�reason�receptor�
raise#r~quiredq
quire)%questionable�questionn
questY
quentuquency;quantify�	quan8quality|	qual�!quadriceps?quadricep�q	�put	Opurposepurely�pulley�pulldown�	pull�'pubmedmedlinePpubmed�publish�#publication�ptk
�	ptenpsk�prudent�proximityeproximaldprovide �protocol �#protein·kg
j)proteinmatched�proteinkg�protein�prosperoB#prospectiveAproposeq   	properesistive�rad·s−�!rads·s−�reversal'psychologicalxrecoverl%satisfactoryksallesjregainhrhomboidsLsartorius:say{satellite�%sarcoplasmicb!sarcoplasmCsarcomere^!sarcolemma=
sarcoWsar�santos
sample�safety�	safe�s.running	�run�	rule�rotaterom)	role_rodent
nrobust�roberts�rmP	rize�	risk �	rise�rir2rie�ribosomepriboso"rewrappe	Qreward�
reviewreveal�return=reticulumY
reticRresultant�result �#restrictionN	restr!responsive�response*responder,!respondent�respond�respon�%respectively�respect	�respeclresonance�/resistancetrainedA!resistanceresistresidue;residualNresident reservem!researcher �research/#requirementrequireZrepresent�reporting8report�!repetition%repeti�	repe�repairrep�	rentPremoval	Hremedy2remains�remain �	relycreliableerelevant
 relevance�releaser	Greleaseg	rele"!relatively�relative�%relationship relation,relateH	relaFrel�reinforce&
reima	:regulator
t!regulationregulate{regularly%!regularity	fregular%registration�register:regionals!	regimentedregard2!refutationI!refinement�refinedreference{	referiref
�reduction�reduce�
recur%recuperative�%recuperation�rectus##recruitment`recruit/)recreationally	recoveryi
recov2)reconciliation�reconcile�)recommendationM!recommendahrecommend*recently �recent3receive�rec�reasoning�!reasonable`reality�readjust6reader!	read �reaction+
reachfrdrb�
rauchrationale�ration{
ratio�rating�	rate�ratErapamycin	�	range'randomlygrandomizeqrandom	�
   ���������{k[L@2&	����������}qg^NB1"����������xgS>*���������yncV@8, ���������ul_TH;2'
�
�
�
�
�
�
�
�

s
e
U
L
@
/
!

	�	�	�	�	�	�	�	�	�	�	�	r	g	]	P	E	;	0	"		�����������~tj^RF8-#���������{rg]I8,  ���������ymbWK@1%���������ocXM=1&��������ymaWMA4$���������vkVJ<."�������kXE3&   shepstone�stretchP%stressrelateC'stressmimickeg'stressinduced�+stressassociate
Jstress�%streptomycin�'strengthpower�-strengthoriented�3strengthhypertrophy�-strengthhypertro�!strengthenFstrengthzstreaming�strcng�stratify�strategy �strain�+straightforward
storewstorage�	store	stop�stitutes|+stimulussensing�+stimulusfatigue�stimulus1#stimulationstimulate�stiffness
�
stiff
�steroidf!sternlicht^sternal1stepup{	step�	stem �steele�statusD!statistics�'statistically/#statisticalstatement�
statestarting!standpoint �%standardized�#standardize�standardF!standaloneX
stand �stance �!stagnation	q
stage �
stack	?stable�!stabilizer�'stabilizationGstability`sta�squattingF
squat�sprinz�sprint�spreadg)sportsmovement%sportsdiscus�sports�
sportwsponse�splitting�
split�splice
w
spite	A
spine�spinae
spike�
spend7	spegx
speed�#speculative�speculatespectrumk%spectrometry�
specta#specificity�%specifically �specific �specifdspecie#specializedT)specialization&	specw
speak�spe�spatially�sparingly�#spangenburg�	spanQ
space@soy�#southampton�source	�
soundI%sorptiometry�soreness�'sophisticated�	soonj
sonal
7somewhatjsoleus�solent�solelyd	sole
B	soft
�sociation/sociate7social�socalled+snp6smp
3smooth�
smith�smilios	smds�smd�
small�!slowtwitch�	slow �slightly
sleep�	slca
slackgskinned
�	skim�
skillskeleton
�skeletal�
skele/	skel5sk
�size	situationsition	sitecsitate�sit�+sistancetrainedE)siondetermined�	sion�singlenu3#singlejoint�single�sin�!simulation:simulate�simply �simple�similarly �!similarityCsimilar �	simi�sim�'significantly#significant)signalregulatesignaling
PsignalB	sign�
sightDsig�sidered�sidere 
sider	eshuttleoshowshortterm�3shortermusclelength�'shortermuscle�shortenf#shortcoming4
shortu
shift�shibataDsheathe1shearing�
shear
�sex	�severe
�setend�set_-sessiontosessionsession�
serve�
serumBseriouslyvseries�serial�sequester�sequence
separateMsensus�sensor
Gsensitize 'sensitization�sensitive�
sense
senna�)semitendinosus�+semimembranosus�)selfdetermined5#selectorize	T#selectivelyhselective�selection�select
selec�	seep�seeminglyH	seekisee�sectionalVsection �secretion{secrete�secondary�#secondarily�second�!seccentricJsec�	seat�season	�searchOse�scrutinya
scrumscriptionscopic�sclerosis
�!scientificvsciences�science<
scien�schwannD!schwanbeck�schwabschumann	�school�scholarly �!schoenfeld �!schieppati�scheme_schematicyschedule	�scenario�scavengerTscatter5scarcity/scarce �scapula-
scanto   �� ����������seZL>+!������yjPE:+
��������qaRH;-	���������|qcWI;+��sjWK;0	
�
�
�
�
�
�
�
�
�
s
d
W
K
:
0


	�	�	�	�	�	�	�	�	�	{	o	a	Y	K	>	4	)			���������teYK?0#��������k\A5*���������{mbT@7*�������tfW?6*!���������}sk]I4*�������ulcXJA/ �������zl]SI8"��   � %collectively� situation� ankrd� #latencoding� -synergistablated� #transporter� slca� pten� inhibitor� suppress� 1exerciseassociated� +differentiation� 'transcription� tead� !coactivate� cofactor� +transcriptional� -mechanosensitive� wwtr�
 taz�	 paralogue� %yesassociate� yap� effector� plcγ� cγ� pip� %bisphosphate� 5phosphatidylinositol�  !conversion� stiff�~ !attachment�} 'phospholipase�| %acidgenerate�{ %phosphatidic�z 'posteccentric�y )phosphorylated�x feed�w +phosphorylation�v )activityrelate�u activate�t sk�s mtor�r tsc�q sclerosis�p tuberous�o !igfinduced�n 3autophosphorylation�m tyr
�l myotube�k cc�j cultured�i move�h tyrosine�g #nonreceptor�f ptk	�e encode�d fak�c 1costamereassociate�b dystrophy�a duchenne	�` severe�_ dmd�^ 1dystrophinencoding�] mutation	�\ normal�[ 7vinculintalinintegrin�Z 9dystrophinglycoprotein�Y costamere�X !associated�W zdisk�V exterior
�U connect�T ilk�S )integrinlinked�R integrin�Q talin�P vinculin	�O anchor
�N onwards�M noncancer�L anchorage�K agar�J soft	�I cancer�H discover�G %historically�F +costamererelate�E %transmission�D #filamincbag�C titin	�B modify�A 7hypertrophytriggering�@ laterally�? )longitudinally�> -forcetransducing�= bone	�< tendon�; transmit�: striated�9 value
�8 µncell�7 !fibroblast�6 /actincytoskeleton�5 pn	�4 myosin�3 µn
�2 skinned�1 nonmuscle	�0 unique�/ surround	�. matrix�- stiffness�, #compression�+ #deformation�* shear�) 'mechanosensor�( withstand�' %cytoskeleton�& skeleton�% being	�$ wonder�# organism�" ms
�! gravity	�  evolve� earth� life
� capable� opposite� wk� workload	� caveat� synthetic� !fractional� label� -timeundertension� yr� ±� near
� address	� little� /hypertrophyinduce� link� !metabolism� #confounding� host�
 #castinduced	�	 flexor
� plantar� plantaris� %mechanically� goldberg	� alfred� 
normal
� ref� )immobilization
�  atrophy� intuitive�~ direction�} )exerciseinduce�| act	�{ actual�z %transduction�y event�x !expression	�w splice�v 'mechanogrowth�u igf�t regulator�s '
firstinline
��r �)
initiate
hypertrophy
stimulus
trigger
hypertrophic
signal
transduction
skeletal
muscle
fiber
hypertrophy
response
resistance
exercise
sensor
�q !unanswered�p molecule�o ablation	�n rodent�m blockade�l 
hub
�k 'wt−·day−�j #protein·kg�i diet
�h consume�g inbetween�f exerciser�e ≈�d nutrition�c incidence
�b disease�a mortality�` +diseasespecific�_ allcause
�^ million�] half�\ longevity�[ !metabolite�Z /exerciseregulated�Y probably�X adhesion�W focal�V !equivalent�U !costameres�T 5deformationinitiated
�S nuclear�R )mechanosensing�Q %incompletely�P signaling�O autophagy�N hippo�M mtorc�L 5filamincbagdependent�K candidate�J +stressassociate�I 7damageinjuryassociate�H 5
hypertrophy
sensor
	�G sensor�F 9
hypertrophy
stimulus

�E elusive�D 9hypertrophystimulating
�C cascade�B sole�A 'mtorcmediated�@ striking�? exerkine�> issue�= canadian	�< patent	�; submit
�: nancial�9 nonfi�8 fee�7 sonal
�6 council�5 dairy�4 grant�3 smp�2 )fitnessrelated�1 ton
  �Ag�+$��;w���������ypf]SD;/$		������!���zp�dZNC8,�  ����������~tgWE;+
�
�
�
�
��V
�
�
m
ZR
A
5
'	


	�m	�	�	�	�	�	�	�I	�	�	p	_	M	;	%��������y��n]�TH9-.#[������������+}�~siaVNB:�1# �������~�pdV�GA9-������������r_Ti`jWsOE:'No��������sj_M9,
������������I8(������   !microcycle�#multiplesetmigrate�'macromolecule�mu�/mechanochemically�+mediumintensity�maimum�marked�%mobilization�mitigate}!modulationxmagnifyqmheYmgfNmechanoM!modifiable9mitosis8module	mapk-mitogenactivated#molecularly1mechanostimulationmrf		mrnamitotic�#mitotically�#miscounting�morgan�	lynn�%maximization�	matt�mvcTmutation
�+musculoskeletal!#musculature(#muscularity�muscularg)musclespecificr%musclebuilde`
musclemus,)murfproteasomelmultitude3multiset�!multiscale�multiple')multinucleatedP!multilevel�!multijointmultihip�1multicompositional�
multi�'mtorcmediated
A
mtorc
M	mtor
�ms
�mri	mps�mpb�movement;	move
�
mouse8
motor�
motion(
motif�mortality
a!morphology6'morphological$moreso
month�'monophosphate	�momentum"momentaryfmoment�molecule
pmolecularmodulate!modify
�modifierOmodest�moderator�!moderation�)moderatevolumeL%moderateload}moderate'
moder~1modelingsimulationmodelingB
model�	mode modality�mod�mo�mmolkg=mmw
mlineTml�	mizeB
mixed�mix�'mitochondrialQmitminute�minority
minimumminimize �minimal	ming#	mind�min�	milo�million
^	milk�	mild�midway�middlemanmiddleUmid #microscopic�
micro�metric�#methodology�-methodologically�)methodological�method �	meth�!metabolite
[!metabolism
�metabolic�mediate�'manifestation}-metaanalytically1%metaanalyticU%metaanalysis �#metaanalysem!mesocycles
$mesocycle	�merrigan	mention �	mentm'membraneboundNmembrane<melbourne�	meetC!mediumterm
mediumEmediator8mediation=medialis:medial�3mechanotransduction�'mechanosensor
�-mechanosensitive)mechanosensing
R'mechanogrowth
v+mechanistically#mechanistic�mechanism�%mechanically
�!mechanicalmechan�#measurementmeasure
!measurable�	meas�%meaningfully!meaningfulhmeaning!	mean�#mealinduced�	meal�mcmaster�mckendry�mcbride�mc�maximus�maximum
maximizemaximally�maximalW	maxiAmaxQmatter�matrix
�
matic	v
matchgmasterI	massmarker^	markmarily	�!marginally>margin�marathon�!manuscript�%manufacturer�
manual~manner �%manipulation[!manipulate2manifest�mandingu
mancedmanageYman'mammalian	�	male�mal#	make�mak
+majority�
majorhmaintainDmainly�	main �magnitudemagnetic�#macroscopic�!macrocycle	�%machinebased�#machinebasewmachine�m�-lysophosphatidicYlyS
meritu
moverS!movementsaQ!multiangleF#multiplanarE!motivation'luteinize!myonucleusmyonuclei�!myonuclear�myoneural�myokinemyogenin%myogenicallysmyogenic�!myogenesis#myofibrillo�%myofibrillar�milieu�!lymphocyte�!macrophage�myofibrilImyofiber�
myofin	myodmyocytemyoblast�myfmw�	mvps]#microtrauma�   �� ��������}t`PD6"	�������mbTA4!����������vj\H6"���������k]QH>1#���������ym_UE7*
�
�
�
�
�
�
�
�
�
{
s
d
S
I
>
/
$
	�	�	�	�	�	�	�	�	�	x	m	b	G	=	/		������xhSH;-�������oZMB1$	�������}qcUH<.����������uj[L>/!���������~mcTG5$��������p`TH7, ������teXD;)	��           	� oxygen�
 reactive�	 drug� -antiinflammatory� %nonsteroidal� aid� )cyclooxygenase
� myokine� substance� enter	� immune#�  Moxidemetalloproteinasehepatocyte	� nitric�~ !stretching�} quiescent�| -damageassociated�{ #susceptible	�z repeat�y #eimdrelated	�x remove�w month�v #celldeplete	�u expand	�t derive�s !presumably�r 'proliferation�q #proliferate�p nondamage�o %nearlifelong�n recipient�m +fiberassociated�l !transplant	�k injure�j #regenerated�i uninjured�h injection�g #cardiotoxin�f %regeneration
�e cooccur�d virtually�c ampk�b #excessively�a %longduration�` %trainability�_ run�^ marathon�] soreness	�\ plasma�[ !cumulative�Z pretraine�Y flann�X !connection�W )eimdassociated�V fascicle�U parameter�T necrosis�S pervasive�R continuum�Q hubal
�P hyldahl
�O secrete	�N escape�M creatine
�L disturb�K %inflammatory�J local�I streaming�H zline�G #microscopic�F 5
repeat
bout
effect
�E -resistancetraine�D lengthen	�C global
�B abolish�A putative�@ knockout	�? hamper�> 3mechanotransduction�= frey�< property	�; reason�: 9httpsgtexportalorghome�9 #loadinduced�8 !gadolinium�7 %streptomycin�6 vivo�5 #nonspecific
�4 mcbride�3 #spangenburg�2 piezo
�1 channel�0 +stretchactivate�/ !myonuclear�. !yapinduced�- /deformingyapmtorc�, 'filaminbagyap
�+ nucleus
�* cytosol�) #translocate�( %intriguingly
�' passive	�& expose	�% desmin�$ %intermediate�# filament
�" tubulin�! thick�  +dephosphorylate� !completely� Cfilaminbagmtorcyapautophagy� +stimulussensing� !bagfocused� )aforementioned� #contracting� #phosphatase� 'highintensity� -phosphoproteomic� %synaptopodin� normally� away� 7hypertrophyassociated� motif� acids� #tryptophane� dimer� receptor� androgen� !powerpoint� )figuredownload�
 download�	 #degradation� casa� !autophaghy� /chaperoneassisted� synpo� myonuclei� amotl� ppxy� #prolinerich�  sequester� ww
�~ intense�} deformed�| !zdisklinke�{ reference�z text�y schematic	�x attach�w #actinlinked	�v deform�u homodimer
�t vshaped�s /actincrosslinking
�r filamin�q localize�p !myopathies�o flnc�n filaminc	�m domain�l )murfproteasome�k !convincing	�j unfold�i actually�h %consequently�g slack
�f shorten�e actin�d #actinmyosin�c #terminology�b +exerciserelated�a numerous
�` binding�_ bind�^ atp	�] pocket�\ atpbinde�[ -stretchactivated�Z passively�Y !elasticity
�X elastic
�W portion�V 'ibandspanning	�U middle�T mline�S myopathy�R #titinencode�Q giant�P ttn�O exert�N )postresistance�M %cytoskeletal�L )differentiated�K isotonic	�J uptake�I glutamine�H #hypoosmotic
�G culture�F bring�E rat�D %interstitium�C definite�B edema�A eimd�@ temporary
�? briefly	�> 
pump

�= applied�< 'costamerebase
�; residue�: 'phosphorylate�9 wildtype�8 mouse�7 %overexpresse�6 itga
�5 isoform�4 %αβintegrin�3 1synthesisstimulate�2 )identification�1 5hypertrophymediating�0 effectors	�/ loaded�. dgkξ�- kinaseξ�, )diacylglycerol�+ reaction	�* locate�) !generating�( )acidsynthesize�' pld�& #zdisclinked
�% butanol�$ !inhibition�# anterior�" tibialis�! modulate�  sensitize� abundance
� unknown
 �� ���������xi^N<-!���������vk[I:)��������}pg\G0"a����������p�ygYM</"	
�
�
�}
�
�
�
�
�
�
~
h
W
B
7
(

 	�	�	�	�	�	�	�	�	�	{	q	c	Y	P	E	:	,	 		����������}oe\K@2(���������vfVKA4(����������tiZOD8"�����������{hYJ4(����������xjaVKB8-���������t_R?*�������{i[K<+������   #supposition>trauma#trapezius+#transportertransporto!transplant�transmit
�%transmission
�#translocate�#translation�%translatable�transient�!transgenic�%transgenesis�+transferability%transduction
ztransduce)transcriptomic1+transcriptional'transcriptiontralize�+traininginduced|trainingtrainee%trainability�
train@'traditionallyi#traditional�tradition	
tradi�tradeoffctractile[#tracellular�tra�totality2
totalt	tory�tor
topic �
tonal�ton
1toleratextolerable0todate	i
tocol�tivelym	tiveptition�#titinencodeR
titin
�!tistically�tissuetionshipGtional�	tion\tinued	
tinely�-timeundertension
�timescale	[timeframeD'timeefficient�timeeffi	�	time6	tilt	>
tight}tified	{
tifie	�!tification9
tific�ticipant	�tic�tibialis")thresholdbased�threshold�-threedimensional�threat�
thinkr
thighGthickness;
thick�thesisJthereof	�	ther	+
theory@theorize\'theoretically#theoretical�	theo�theless8theiuscax
thank�th	textz%testosteronettestis`testing�	test�
tesch<!tervention	K!terspersed
#ternatively'#terminologycterminate�termmtenuated	�tensity�)tensioninduced�tension�
tenet�tendon
�	tendtempos �temporary@	tempo&templateC
temic�#tematically�tem�	tein�tegrates!technology�technique#technicallyUtechnical	W	tech�tearing�	tear�	teadtaz
tax�tation	�%taskspecific	tasktarget �tapering�
taper
talin
�tal0	take}takarada�	tainvtachment�
tableSt{systemicy)systematically�!systematic �systemsys�synthetic
�!synthesize1synthesisstimulate3!synthesise�synthesis�
synpo�+synergistically�#synergistic�-synergistablatedsynergist�synergis�syndrome!%synaptopodin�syn	�symposium�#sympathetic�switch�swelling	
swell�	swaysustained~sustainbsuspendF#susceptible�survey�surround
�%surprisingly�!surprisingsurgery#suppressionusuppress!supposedly�!supportive�support �supply$+supplementationsuperset�#superiority�superiorw/supercompensation4+supercompensate
super	sup�summaryAsummarize#suggestHsug�%sufficiently`!sufficienty%successively	�!successive	 %successfully�!successful	�substratex'substantially#substantial�substance%subsequently	�!subsequent�
subset!suboptimalYsubmit
;!submaximal�!subjective	nsubjectpsubgroup~#subdivision�subdivide,#subanalysiss!subanalyse4su�
study �stuart�structure%structurally@!structural
struc&strongman�strongly�strongUstringSstriking
@strictly
'strictlenientX
strictPstriated
�!stretching�stretched�-stretchactivated[   Fstretchactivate�stretchP%stressrelateC'stressmimickesuperslow~surfaceYsymmetryV
suraeJ'triangulationE
trial�!tremendous�treatment	�treadmill�traverse�!supinationB   �� ��������sfRE6 ����������s���������wj[K<,�����������qfTF6*��������rdUF9#
�
�
�
�
�
�
�
�
�
|
m
Z
K
@
6
+

	�	�	�	�	�	�	�	�	�	�	y	i	��������rcQ>0 �������zbL</ �������|sc[MD<3(����������yj]QI?6*��������ucRF8,���������}qfWJ?."��������q`SB3'��          	�w arouse�v 5
myogenic
stem
cell
	�u lamina�t basal	�s reside�r apoptosis�q #replacement
�p undergo�o #postmitotic�n elongate�m #arrangement�l intricate�k #miscounting�j erroneous�i avian	�h kelley	�g partly�f 'nonfunctional�e !endomysial
�d fibrous�c !perpetuate	�b belief�a #concomitant�` count
�_ incline�^ treadmill�] climb	�\ morgan�[ lynn	�Z serial�Y diameter�X myogenic�W chain�V myofiber�U %perturbation
�T enlarge�S #contractile�R 3hypertrophyspecific
�Q dictate�P heighten	�O gender�N dominant	�M couple�L #nonexistent�K %maximization�J full	�I aspire�H %recreational�G quantity�F !competitor�E vital�D football�C #extensively
�B twofold�A !impressive
�@ lengthy�? routinely	�> fairly	�= pursue�< res�; cond�: bj	�9 advice
�8 helpful�7 alexander�6 matt�5 chen�4 ju
�3 ylänne�2 jari�1 thank
�0 finland�/ #jyväskylä
�. october�- symposium�, )acknowledgment
�+ approve	�* revise�) edit�( draft�' jjh�& ml�% dlh�$ prepared�# hw�" !disclosure�! quo�  )experimentally� big� %conclusively� 'falsepositive� )interpretation� feasible� %spectrometry� +exercisetrained� 3coimmunoprecipitate� proteomic
� mediate� !physically� -hypertrophysense� 1hypertrophysensing� inducible� %transgenesis� !transgenic� 'sophisticated� !usefulness
� problem� evaluate� /pharmacologically�
 knock�	 !conclusive� +synergistically� interact� 'damagerelated� %
sweet
spot
� interfere� )forceproducing� hermetic� certainly
�  realize� %growthrelate�~ redundant�} 'manifestation�| +traininginduced
�{ related`�z �E
mainly
zdisk
costamere
mechanical
hypertrophy
stimulus
sense
transduce
resistance
exercise
�y !filaminbag�x speg�w obscurin�v robustly�u contract�t hub	�s zdisks�r )musclespecific
�q generic�p partially	�o potent�n prolonged�m %energystress	�l ampkα�k inoki�j soon�i aicar�h activator�g 'stressmimicke�f evolution�e 3phosphofructokinase�d 'phosphorylase�c flux�b rheb�a gapdh�` !glycolytic�_ hek	�^ triple�] mvps�\ theorize�[ #halfmaximal�Z cellbase�Y -lysophosphatidic
�X placebo�W larginine�V increased�U drinking�T scavenger�S nitrogen
�R citrate�Q +αketoglutarate�P +anabolismrelate�O modifier�N 'membranebound�M %gprdependent�L ohno�K )lactaterelated�J l	�I brooks	�H george�G !laboratory�F caffeine�E %lowintensity�D vitro�C %stressrelate�B serum�A vague�@ 
classic
�? nonsteady�> woman	�= mmolkg�< tesch	�; biopsy�: logic�9 fatigued�8 hydrolyze�7 #resynthesis�6 delivery�5 !hydrolysis�4 biomarker�3 !glycolysis�2 adp↔atp
�1 lohmann�0 adp�/ %continuously�. oxidative�- %resynthesize�, !hydrolyzed�+ )nonsteadystate	�* refute�) #powerlifter�( eliteq�' �g
metabolite
simply
augment
muscle
activation
cause
mechanotransduction
cascade
large
proportion
muscle
fiber
�& proposal�% da�$ normoxia
�# hypoxia�" %intermittent�! ph�  pcr� +phosphocreatine	� regime� !invariably� mark� vascular� !compensate� -occlusionrelated� occlusion
� occlude� )intermittently
� patient
� surgery� +braceimmobilize� 'postoperative� upstream� middleman� +supplementation� #antioxidant� ros	� specie
N) ��������~rh]PE=,#����D6+	����������~seUE0�&
��
�
�
�
�
�
��
�
x
h
U
F
4
#
�
	�	�	��	�	�	�	�	�	�	{	m�	\	O	=�	.	 	�	�����~�������sj]OZF;-	���������|pcYpH�=1%�
������������wj^TF8��O���������wnf�ZMB9/�&
�����������}gYOC:- �����e������xe�[OE:1�'
 �����.���������:ph_VMD�;2)±µn
�±
�   ≥�≤z≈
e−r’�—�–Rτ	�%αβintegrin4×|µncell
�µn
�±
�°		zone\	zinc>zealand�
zdisk
�#zdisclinked&yr
�
young�	york�
yield[%yesassociate	year�yap
xweek		xray�xforce	=	wwtr'wt−·day−
kwriting�
worth �workout�workload
�	work`	word�wonder
�	wolfvwk
�withstand
�'withinsubjectE/withinparticipant		wishHwinwood	�wingate�win�wilson	�	wilk �wildtype9-widthorientation �
width�widely�	wide �wholebody	lwhilst�
wheth#welltrained	Jwelldevel)	well�weightb!weektoweek
!weeksmonth~weeklyweekb	weakway�
water	warrant.walker	Evsvor	2volved	�
volve�voluntaryS%volumematcheM!volumeload�'volumeequatedK%volumeequate�+volumedependent�
volume]!volitional�vol[7vinculintalinintegrin
�vinculin
�vigotsky�	view;vieiraLvidualsFvidual�victoria�	vice	vic�viable�vertical�
versusQversion�versial>
versa
vention	Yvelopment$veloped(velocity�vastus9vastly
vasti�#vasculatureK	vary �variety�varied�
varievariationSvariant;1variancecovariance%variance�variable #variability#	vari	�vantage�vanced	}
vance#
value
�validityyvalidate6
validkval	�utilizedTutilizenute�usually �ushaped�	user	7useusa�urement�ure<uptakeJupright%upregulation�upperlimb.
upper�update)!unweighted	�unwanted]untrainedE%unresponsive�unravel#unpublished�unlikelydunknown!university�#universallysuniversalI	unit\unique
�!unintendedZ	unin�!unilateral�uniform�%unidentified9undulate
#undoubtedly%undeterminedundertake�'understanding�!understand�%underpredict9underpin2)underdeveloped$underde'#uncontested�unclear^#uncertaintyQuncertainzunbiasedS!unanswered
qunjume\	ulumS+ultrastructural�!ultrasound�!ultimately�ultimate�
ulatexuk�ufacturer	@ued	_ualizeual	�tyrosine
�tyr
�typicallytypicalP%typespecific�type$ty%tweensubjectJ!tweengroupT
tween�turnover�	turn\* dturetuberous
�tuation�ttnPtsc
�try
trunktruncat@
truly@	upf!veragarciacunstableX%wellaccepted(triconly	 tricepsztricep		tric	htributory1   [tributing'triangulationE
trial�!tremendous�treatment	�treadmill�	vast	vein�ttubule�unload�∼�)ultrastructure�weaklyjunboundh	turetubulin�tuberous
�tuation�ttnPtsc
�#tryptophane�try
trunktruncat@
truly@	truettrophy trophic 
troll
,trivial�triple^trindade	%triglycerideztrigger�!upregulateQundergo�
vital�twofold�ylänne�!usefulness�zdiskss+αketoglutarateQ
vitroD
vagueA
woman>vascularupstreamuninjured�virtually�
zline�	vivo�!yapinduced�ww!zdisklinke|vshapedtunfoldj   �� ��������wg]K<,#��������mcTG>3$���������n_NF5+���������naQD<1"����������xm]PG=+
�
�
�
�
�
�
�
�
�
l
\
Q
E
9


	�	�	�	�	�	�	�	�	�	t	`	Q	@	3	"		��������{l[I>,	����������paPA2$
������}paS:.���������k_RD5'�������~n`XL<0$ ���������raTG<-���������paS@/#�             �l 'growthoriente�k #freeradical	�j milieu�i #hypertophic�h 'stressinduced�g ischemia�f phosphate�e inorganic�d hydrogen
�c buildup�b secondary
�a impinge�` junction�_ myoneural	�^ debris�] !lymphocyte�\ !macrophage�[ #microtrauma
�Z migrate�Y !neutrophil�X infection�W liken
�V opening
�U tearing�T #homeostasis�S !disruption
�R ttubule�Q shearing�P #lengthening	�O region�N !supportive�M tear�L 'macromolecule
�K exhibit	�J rabbit�I prado�H !slowtwitch�G pkc	�F camkiv	�E camkii�D !calmodulin	�C firing�B mu�A coupling�@ !excitation�? amplitude�> /extramyofibrillar�= /mechanochemically�< #translation	�; unload�: pronounce�9 !generation�8 #responsible�7 !initiation�6 endocrine�5 ischemic�4 hyperemia�3 myoblast
�2 cardiac	�1 smooth�0 )hypoxicinduced
�/ hypoxic�. clearance�- +mediumintensity	�, maimum�+ ∼�* !protective	�) kubota�( bed�' takarada
�& reflect
�% attract
�$ storage�# anaerobic	�" influx�! 3oxidativeglycolytic�  strongly� aquaporin
� osmotic� #contributor� Aalphamethylaminoisobutyric� 1integrinassociated� hydrated� )ultrastructure� 'reinforcement� integrity	� threat� pressure� simulate� hydration� myotrauma� #conjunction� replicate� !administer	� marked� postulate� spike� halflive�
 kda�	 sleep� pulsatile� gland� pituitary� 1autocrineparacrine� !nonhepatic� 'incorporation� %mobilization� agent�  %repartitione� #polypeptide�~ sustained�} mitigate
�| elderly�{ secretion�z !fasttwitch�y fibertype�x !modulation�w !compromise�v seriously�u #suppression	�t commit�s %myogenically�r #replication
�q magnify�p #chromosomal�o transport�n )conformational�m cytoplasm�l %disassociate
�k rapidly	�j weakly�i %biologically
�h unbound�g globulin
�f steroid
�e albumin
�d adrenal�c ovary�b axis�a Ehypothalamicpituitarygonadal	�` testis	�_ leydig�^ nerve�] !regenerate�\ -neurotransmitter�[ %considerable�Z 1cholesterolderived�Y ltype�X donation	�W fusion	�V igfiea
�U locally�T 'differentiate�S paracrine�R autocrine�Q !upregulate�P %
kick
start
�O igfs�N mgf
�M mechano�L !familiarly�K igfec�J igfeb�I igfea�H igfbp	�G igfbps�F circulate�E liver
�D schwann�C !similarity�B name
�A peptide�@ %structurally�? enhanced�> premature�= dismissal�< overt�; gh
�: regimen�9 !modifiable
�8 mitosis�7 #proteolysis
�6 possess
�5 insulin�4 !inhibitory�3 leukemia�2 #interleukin	�1 hepato
�0 damaged�/ !likelihood
�. elevate�- integral�, cytokine�+ #cndependent�* gata�) #caregulated�( cn�' #calcineurin�& implicate�% #cadependent�$ -calciumdependent�# rapid�" jnk�! !nhterminal�  cjun� erk
� kinases� )signalregulate	� module
� myocyte� redox� 'proteinkinase� catabolic� nodal� akt
� aktmtor� calciumca� mapk� -mitogenactivated� %aktmammalian� transduce� #molecularly� 1mechanostimulation� !myogenesis� promoter� dna�
 sequence�	 mrf� myogenin� myod� myf� !regulatory� coexpress� %proportional� mrna� !myonucleus�  !capability
� mitotic	�~ retain�} #mitotically�| ?nuclearcontenttofibermass�{ extra	�z donate�y precursor�x fuse   �� ��������|eXB.���������n[NC3)����������jUK@1  ��������zi[L=0$��������whZI=, 
�
�
�
�
�
�
�
�
~
s
d
U
G
9
(

	�	�	�	�	�	�	�	�	�		q	b	R	F	7	-		����������xk`TI7#���������wgTH;,�����                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                � %overreaching� !microcycle�
 resistive�	 #dynamometer� 'forcevelocity� rad·s−� !rads·s−� shepstone
� gastroc	� soleus� 'derecruitment� !schieppati
�  nardone� reversal�~ superslow�} !heightened
�| cadence�{ blunting�z izquierdo
�y burnout�x 'psychological�w !performing
�v linnamo�u merit�t )concentrically�s %postadaptive�r fifth	�q buresh�p bodys
�o shuttle	�n buffer
�m density
�l recover�k %satisfactory	�j salles�i )counterbalance	�h regain�g category�f up
�e oblique�d abdominis�c !veragarcia�b abdominus�a ball�` stability	�_ crunch�^ !sternlicht�] exception�\ diminish�[ behm�Z anderson
�Y surface�X unstable�W %architecture�V symmetry�U imbalance�T !precedence�S mover�R #inefficient�Q !movementsa�P dozen�O #necessitate�N coverage�M postural�L rhomboids�K abdominal�J surae�I abductor�H #dynamically�G 'stabilization�F !multiangle�E #multiplanar�D nonlinear�C centrally�B !supination�A %partitioning�@ insertion	�? origin�> #supposition	�= branch�< #inscription�; gracilis�: sartorius�9 )branchsuggeste�8 1componentsdistinct�7 inactive�6 adjacent
�5 scatter�4 #inclination
�3 deltoid�2 !clavicular
�1 sternal�0 !pectoralis
�/ depress	�. abduct
�- scapula�, subdivide�+ trapezius�* leverage�) #compartment�( %wellaccepted�' !motivation
�& reduced�% cognizant�$ !highvolume	�# trauma�" !repetitive�! syndrome�  cortisol� luteinize� #chronically� #overtrained� cessation� taper� plus� +supercompensate� -
rebound
effect
	� enable� mh
� smilios	� fourth� !completion	� schwab� lowvolume� #multipleset� %highervolume� 'consecutively� vast� #impractical� bear�
 !intriguing�	 girth
� profile� 'applicability� )fatigueability� 'theoretically� osmolyte� byproduct� gradient� %interstitial�  capillary� seep
�~ deliver	�} artery�| !compressed�{ vein�z !remodeling�y )tensioninduced�x -glucosephosphate
�w glucose�v /bodybuildingstyle�u !prevailing�t evoke�s !inadequate�r %artificially�q prove�p classify�o #customarily�n #sympathetic	�m acidic
� � ���������j�vj]QC7-����������{pe[PG:1%����������tf[MB6'���������q`N:)
�
�
�
�
�
�
�
�
|
m
[
N
A
7
'

	�	�	�	�	�	�	�	�	�	~	r	d	U	D	9	1	)		
w���������xlaQD<-!�����������ylbOB4$	���������y�kb�UE8+���������vj[X=- �������}qj^TJA4)���������p�dWH>2(������������zm_O?/"!!neutrophil    !neutrophil�pereira+!perceptual	o!perceptionu!percentageOperceive�peptideApeople8pendent	�
pende%peerreviewedG
pedal	9pearson=	peak�pcr 
pause�paucitynpattern<patroklos�patientpathway�	path	�patent
<	past�passivelyZpassive�partly�%particularly�!particular!#participate	�#participant>partiallyppartialo	parexparameter�paralogue	parallelEparacrineSpar	�
paper �
pande�pallare)pak~	pair�	pain�pacity~p�oxygen%Moxidemetalloproteinasehepatocyte 
oxideP3oxidativeglycolytic�oxidative.overview%overtraining_!overtraineovertrain	�
overt<oversim	�!overshadow�=overreachingovertraining	roverreach/overly	�!overloaded	.overload�overlap	�#overheadarmnoverheadp%overexpresse7overcome�overall7
ovarycoutward�outside?outputYoutpaceoutline �outcome �!ostensibly	�osmotic�osf>originalDorganize	�%organization
'organizaoorganism
�organelleU	orgaG
order �'orchestration �#orchestrate&option�optimize�optimal �!opposition�opposite
�opinion�'operationally�	oper�opening�	open;ope*onymous	�onwards
�ongoing�one�#onciliation�	omni	6ological�old&	ohnoLogeneity	�
offer	�!odological�odized
odization
#od	|october�!occurrence	�
occurZoccupyL-occlusionrelatedocclusionocclude!occasional	xocs!obtainablekobtainRobserve#observationCobscurinw!obligatory@objective�oa�ny�
nuzzo7#nutritional	�nutrition
dnumerousanumber�num�nucleus�?nuclearcontenttofibermass�nuclear
Snuanced#nuance�nsca}novice�novelty	�
novel+notwithstanding �notion�!noteworthy	]	note �notably�notable	a%northwestern�norrbrand	Snormoxia$normally�normal
�nonvaried�!nonuniform�#nontraining	#nontargeted|nonstrict	%nonsteroidal)nonsteadystate+nonsteady?#nonspecific�)nonsignificant	\%nonresponder-#nonreceptor
�'nonperiodized	�nonmuscle
�!nonhepatic�'nonfunctional�
nonfi
9!nonfailureM#nonexistent�#nonexerciseR#nonetheless�nondamage�)noncontractile�noncancer
�nology	`
noise-nogueira
nodalnitrogenSnitric�nition�
nique�nippard�	ning�nificant�!nhterminal!
nhout�new�!neutralarmm-neurotransmitter\neuron�'neuromuscular�neural_neu�network�net�
nerve^	nentj
nelleH
neity�!negligibleg!negatively>negativesnegateInegFneed0necrosis�necessity	tnecessary�#necessarily�
neces�nearlyV%nearlifelong�	near
�naïve0nautilus	5nature�nationalynation	�narrow
narrativenancial
:	nameBnal�nan�myotube
�myotrauma�myosin
�myopathyS� cmyopathiesp!myonucleusmyonuclei�!myonuclear�myokinemyogenin%myoge%overreaching�nardone�obliquee#necessitateOnonlinearD%partitioningAorigin?!pectoralis0#overtrainedosmolyte   myoneural�   �    f���r\H-�����~bF0�����thP-�����)����ydJ0

�
�
�
�
q
V
E
9
'
	�	�	�	�	�	�	�	p	[	F	:	.	#	�����{k\PC-����ucL;$������wgTA$ ������pXB/�����k�����oT
�����`C)������u^J                    � !myofibril
ORG� !retic ulumPERSON� !sarcoplasmORG� !tal muscleORG
� facORG� chang esPERSON� %micro scopicORG� mpsPERSON&� Kscien tific research mechanicalORG� -the past centuryDATE� #onciliationCARDINAL� 3preex isting muscleORG� ;ial crosssectional areaORG�
 los ingGPE�	 -evanston il usa
ORG� canadaGPE� hamiltonPERSON6� kauckland  new zealand department of kinesiologyORGG� �environmental science auckland university of technology privateORG� /faculty of healthORG4� galabama sports performance research instituteORG� !birminghamGPE� %fitomics llcORG�  )vic  australiaORG melbourneGPE ~ ?sport victoria universityORGM} �social sciences solent university southampton uk institute for healthORG*| Sbronx ny usa school of sport healthORG{ lehmanORG$z Gdepartment of health sciencesORGy )james steele  PERSONx phillipsORG6w ejozo grgic cody t haun eric r helms stuart mPERSONv james pPERSONu !eric helmsPERSONt %james wrightPERSONs firstORDINAL
r rrsORGq +httpsosfiowdjxgORGp grgicPERSONo pallarésNORPn ;upperbody muscle groupsORGm 'httpsosfiouvdORGl six weeksDATEk an hourTIMEj /stasinaki et al  PERSONi !neutralarmORGh !maeo et alORGg satoPERSONf 'mcmahon et alORGe sixCARDINAL	d alORGc -httpsosfioahjnf
PERSONb secondlyORDINALa threeCARDINAL` 7height metaregressionPERSON_ nonwarmupPRODUCT^ )metaregressionORG] dividualGPE\ #terventionsPERSON[ upperbodyORG$Z Gclusively resistance machinesORGY )dis tal musclePERSONX hypertroPERSONW )betweensubjectORG6V edynamometerselectric goniometerstensiometersPERSONU seriou saPERSONT 'halfrom forceORGS seriORG
R romGPEQ weeksDATEP mvicPERSONO pedrosaPERSONN -werkhausen et alPERSONM #ticipant  lPERSONL 1wgt cmjsprint timeORGK %whaley et alORGJ pallaresPERSONI 1torgluteus maximusPERSON	H rmGPE
G rm
GPE
F mpvORGE u  °  °PERSOND l  °  °PERSONC ticipant
PERSONB kuboORGA %martinezcavaPERSON@ gotoPERSON? valamatosPERSON> oneCARDINAL= %metaanalysesORG< !interstudyGPE; 7sessionsexercisessetsNORP: tidybayesGPE
9 csvORG8 ankitPERSON7 /a few weeks laterDATE6 secondORDINAL5 3prom groupconditionORG4 weeklyDATE,3 Qgradepro httpswwwgradeproorg  thisPERSON*2 Sabstrackr httpabstrackrcebmbrowneduORG1 augustDATE0 'pubmedmedlineORG/ %at least oneCARDINAL. %at least twoCARDINAL- englishLANGUAGE-, Ythe international prospective registerORG+ 3metaanalyses prismaORG* %schoenfeld  PERSON) onesizeORG( %recent yearsDATE' bayesianNORP& %sportsdiscusORG[% �3social sciences solent university southampton uk city university of new york lehmanORG$ ;faculty of sport healthORG# %james steelePERSON" bradPERSON&! Eandroulakiskorakakis james pPERSON  3milo wolf patroklosPERSON %jeff nippardPERSON! Ainstitutional review boardORG +pak mw mc rb apORG
 pakORG only oneCARDINAL -nonstrict strictGPE -exercisespecificORG$ Gexercisespecific instructionsORG 3rt exercisespecificORG yearsDATE! Aexercisespecific techniqueORG %longerlengthNORP -longmusclelengthPERSON 3shortermusclelengthORG %longermuscleORG 1longermusclelengthGPE !only threeCARDINAL kassianoPERSON wolfPERSON  ?thigh crosssectional areaGPE twoCARDINAL
 pereiraGPE	 +pereira et al  PERSON mass lbmPERSON !al  keelerPERSON nogueiraPERSON keelerPERSON wilkPERSON !schoenfeldPERSON 7exercisespecific bodyORG" Cexercisespecific k   O�@   N�   �    u����Lx���x?P�"�	���,���	t�o}Z_	�	I	�a���2�G���5U����$
�	�X��	�7`��B�F�q�����
�
�T���2��}�{��5�	6�����
�
5�x��	(T6-��
�.
>[g��&�G�a-���
�
�
#ei"�I�������
	�U�L�j��_�������	V��m- �5	
�
_o
�S
�%	f<��
+
J
o
��,�'more than one �riboso �#rt programs �1the rd and th week �5the early first week �;longerterm hypertrophic �+daystoweeks mps �postex �!longerterm �	infu �+shortterm hours �=proteinkg bodyweightmeal �	exer �mpb �#acute hours �postrt �sponse �1hyperaminoacidemia �forcepro �-strengthhypertro �imental �/subse quent elbow �	week �un �7the mediation ef fect �a week �
nuzzo �!sociations �cor �)associ ational �ten �%con sidering �%the ac crual �%indi viduals �buckner �ar gued �hu �	bril �=physi ological rationale �roberts �+rt myofibrillar �%hypertro phy �#sar coplasm �
daugh �'rt  jorgensen �#myofibrillo �!myofibril
 �!retic ulum �!sarcoplasm �!tal muscle �fac �chang es �%micro scopic �mps �$Kscien tific research mechanical �-the past century �#onciliation �3preex isting muscle �;ial crosssectional area �los ing �-evanston il usa
 �canada �hamilton �4kauckland  new zealand department of kinesiology �E�environmental science auckland university of technology private �/faculty of health �2galabama sports performance research institute �!birmingham �%fitomics llc �)vic  australia �melbourne?sport victoria university~J�social sciences solent university southampton uk institute for health}'Sbronx ny usa school of sport health|
lehman{!Gdepartment of health sciencesz)james steele  yphillipsx0ejozo grgic cody t haun eric r helms stuart mwjames pv!eric helmsu%james wrightt	firstsrrsr+httpsosfiowdjxgq	grgicppallaréso;upperbody muscle groupsn'httpsosfiouvdmsix weekslan hourk/stasinaki et al  j!neutralarmi!maeo et alhsatog'mcmahon et alfsixeald-httpsosfioahjnf
csecondlyb	threea7height metaregression`nonwarmup_)metaregression^dividual]#terventions\upperbody[!Gclusively resistance machinesZ)dis tal muscleYhypertroX)betweensubjectW0edynamometerselectric goniometerstensiometersVseriou saU'halfrom forceTseriSromR	weeksQmvicPpedrosaO-werkhausen et alN#ticipant  lM1wgt cmjsprint timeL%whaley et alKpallaresJ1torgluteus maximusIrmHrm
GmpvFu  °  °El  °  °Dticipant
CkuboB%martinezcavaAgoto@valamatos?one>%metaanalyses=!interstudy<7sessionsexercisessets;tidybayes:csv9	ankit8/a few weeks later7
second63prom groupcondition5
weekly4&Qgradepro httpswwwgradeproorg  this3'Sabstrackr httpabstrackrcebmbrownedu2
august1'pubmedmedline0%at least one/%at least two.english-*Ythe international prospective register,3metaanalyses prisma+%schoenfeld  *onesize)%recent years(bayesian'%sportsdiscus&X�3social sciences solent university southampton uk city university of new york lehman%;faculty of sport health$%james steele#brad" Eandroulakiskorakakis james p!3milo wolf patroklos %jeff nippardAinstitutional review board+pak mw mc rb appakonly one-nonstrict strict-exercisespecific!Gexercisespecific instructions3rt exercisespecific	yearsAexercisespecific technique%longerlength-longmusclelength3shortermusclelength%longermuscle1longermusclelength!only threekassianowolf?thigh crosssectional areatwopereira
+pereira et al  	mass lbm!al  keelernogueira
keelerwilk!schoenfeld7   W+rt myofibrillar �   V1hyperaminoacidemia �   '    w���������������������|vpjd^XRLF@:4.("E(
�
�
�
�
i
H
-
	�	�	�	�	�	�	d	L	3	������iL-�����nP/����}hI(����lN4����s]D"����iH'����nR1�����fB%                     � growthNNdobjkeywords!� #developmentNNcompoundmuscle� formNNcompoundmuscle� keywordsVBZROOTkeywords�  #
_SPdephypertrophy !variationsNNSpobjversus~ !lenientJJamodvariations} versusINprepstricter| stricterJJRpobjof{ #investigateVBconjfollowedz futureJJamodresearch y +recommendationsNNSpobjwith x +aboveJJamodrecommendationsw !withINprepaccordancev !accordanceNNpobjinu adoptedVBNconjfollowedt followedVBNccompsuggesteds !kinematicJJamodguidelinesr !universalJJamodguidelines q suggestedVBNccomprecommend p thereforeRBadvmodsuggestedo itPRPnsubjpasssuggestedn limitedJJccomprecommendm effectNNconjresearchl theirPRP$posseffect k %aspectsNNSdobjmanipulatingj theseDTdetaspectsi impactNNpobjonh howeverRBadvmodexisting g %biomechanicsNNSconjanatomyf anatomyNNpobjfrome appliedVBNamodanatomyd fromINpreptheoryc theoryNNpobjonb impliedVBNamodtheorya basedVBNconjneeded` generallyRBadvmodbased_ areVBPauxpassbased ^ #patternsNNSconjpositioning ] movementNNcompoundpatterns\ #positioningNNpobjfor[ #bodyNNcompoundpositioningZ !guidelinesNNSdobjenhancesY enhancesVBZccompdetermineX furtherRBadvmodenhancesW phaseNNpobjofV !concentricJJconjeccentricU orCCcceccentricT eccentricJJamodphaseS eitherCCdetphase R %durationNNdobjmanipulating"Q %manipulatingVBGcsubjenhancesP %whetherINmarkmanipulatingO determineVBadvclneededN neededVBNconjrecommendM isVBZauxpassneededL researchNNnsubjpassneededK moreJJRamodresearchJ sVBZconjbetweenI  _SPdepbetweenH betweenINprepemployingG employingVBGadvclneededF alsoRBadvmodemployingE whileINmarkemployingD lengthsNNSpobjatC longJJamodlengthsB atINpreptrainingA !emphasizesVBZrelclrom@ aDTdetrom? employVBccomprecommend> shouldMDauxemploy= onePRPnsubjemploy< thatINmarkemploy; recommendVBPadvclaims: wePRPnsubjrecommend9 romXXdeprecommend8 motionNNpobjof7 rangeNNconjtempo6 andCCcctempo5 tempoNNpobjas4 !repetitionNNcompoundtempo3 typeNNcompoundtempo2 #contractionNNcompoundtype&1 !#kinematicsNNScompoundcontraction,0 -#exercisespecificNNPcompoundcontraction/ asINprepvariables. suchJJamodas- variablesNNSpobjon , !focusingVBGadvclmaximizing+ !maximizingVBGpcompfor* forINpreptechnique) properJJamodtechnique( #constitutesVBZpcompon' #whatWPnsubjconstitutes& onINprepevidence% !evidenceNNdobjsynthesize$ existingVBGamodevidence# !synthesizeVBxcompaims" aimsVBZconjshown! reviewNNnsubjaims  narrativeJJamodreview thisDTdetreview #hypertrophyNNdobjmaximize #egNNcompoundhypertrophy' ##adaptationsNNScompoundhypertrophy maximizeVBxcomptrying tryingVBGadvclaims !especiallyRBadvmodwhen techniqueNNpobjof! exerciseNNcompoundtechnique  #appropriateJJamodtechnique !ofINprepimportance !importanceNNdobjemphasize !theDTdetimportance emphasizeVBPadvclpromote oftenRBadvmodemphasize 'practitionersNNSpobjin engagingVBGcsubjemphasize whenWRBadvmodemphasize sizeNNpobjin muscleNNcompoundsize inINprepincreases
 increasesNNSdobjpromote	 promoteVBx   v�1   t�6   s�;   r�A   o�A   n�I   l�R   j�\   i�^   g�e   e�h   c�j   b�k   `�k   _�n   ]�n   \�n   Z�o   Y�r   U�t   S�q   R�r   Q�r   M�r   K�s   J�s   H�u   G�u   E�t   D�t   C�{   A�   ?�   >�    <�    ;�   8�   7�   6�s J:   mI1J^ 	��Z�������w���5f����b���\�	�2��~0w�-���M=3Z[�&�A�
���_�F �r�	L�T�8)�JbB��M���Zg��	�
�
�
{�)�	q�	���r�V�	�K�!�?$�L�?	X	�
k�1����
#
��	���
G�vl	�P�O��U	.�m���	 ��� w�\FF?�?v�d�	|	��&����P�L�!�"X	=x�P
�����	�	��Ke��jo��	�4�/x�4���^�
^���
�ge���G0+�(
�A�������
�n���
8�5
S	�^	g�=o}��
��B�� current!endeavored	date
paperscholarly!comprisingvariousreadingoptimal#extrapolateitself
studysectionmethodsoutlinedexaminingstudies
previous	similarlyambiguitycreatingdefine!explicitlynotdo	theymentioned 	some �referring �authors �inferred �stages �
early �learning �#highlighted �	have �#researchers �depending �	vary �include �%prescription �stands �scarce �outcome �exploring �directly �although �prevent �	well �'effectiveness �emphasis �practice �	both �#prescribing �alignment �'orchestration �involves �injury �	risk �!minimizing �targets �#effectively �ensure �movements �bodily �execution �!controlled �pertains �suggest �!standpoint �	thus �
power �requiring �#potentially �
goals �specific �	’s �!individual �by �affected �heavily �be �can �noting �
worth �position �stance �-widthorientation �	grip �correct �involved �groups �primary �following �includes �checklist �describes �manual �	nsca �#association �%conditioning �strength �national �!components �#constituent �its �!literature �!scientific �!definition �!agreedupon �#universally �no �
there �currently �enhance �may �proposed �an �up �	make �different �describe �	used �” �“ �	term �effective �component �key �referred �variable �another �failure �muscular �momentary �proximity �effort �intensity �	week �
group �per �performed �	sets �ie �volume �certain �%manipulation �requires �
known �%introduction �growth �#development �	form �keywords �
 �!variationslenient~
versus}stricter|#investigate{
futurez+recommendationsy	abovexwithw!accordancevadoptedufollowedtkinematicsuniversalrsuggestedqthereforepitolimitedn
effectm	theirlaspectsk	these   =!chillibeck�   @#constitutes(s� sensorc   ktheses]� 
   9   [	rela�   T%prioritizing
E   pexamplel   hmin,
   F-forcetransducing� 
lines�   xdropset�n 
   BdegreesA   :)betweensubject�   9analyzeJ   I!hydrolysis�� 	-s   ^sensors�   dure�� thanks�   X!perimental}� minimal�9 �    Plinkingf   qnot 
   @   L%instructionsb�'hypertrophies��potential��direct�� +opinion�~    Betc�� 	   T   h)mechanosensors�s��'contradictory
# l!chillibeck�   9#assessments�#       h   Tsupplied�   I%implications�   Ffromd   hkinases �!identifies   [!ultimatelyN� !%powerlifting�� i�   ^'shortermuscle� Ouali   [#uncontestedB   Tsupported9  � 
media�� 
se   dregisterR 4explainedu   L   a-strengthhypertro�l�d	H@  
   ^    Xpractice �r� J   disolate�s 
hi   @!elucidated	+� 'complementary�n�fV��eN�d� 	time =� 	each*Oqss�� re�ejis�e4� doubleg7�      :etAX    �% ���iN5����~]?!����gN0����y\C+����|_E(
�
�
�
�
i
H
-
	�	�	�	�	�	�	d	L	3	������iL-�����nP/����}hI(����lN4����s]D"����iH'����nR1�����fB%                     � growthNNdobjkeywords!� #developmentNNcompoundmuscle� formNNcompoundmuscle� keywordsVBZROOTkeywords�  #
_SPdephypertrophy !variationsNNSpobjversus~ !lenientJJamodvariations} versusINprepstricter| stricterJJRpobjof{ #investigateVBconjfollowedz futureJJamodresearch y +recommendationsNNSpobjwith x +aboveJJamodrecommendationsw !withINprepaccordancev !accordanceNNpobjinu adoptedVBNconjfollowedt followedVBNccompsuggesteds !kinematicJJamodguidelinesr !universalJJamodguidelines q suggestedVBNccomprecommend p thereforeRBadvmodsuggestedo itPRPnsubjpasssuggestedn limitedJJccomprecommendm effectNNconjresearchl theirPRP$posseffect k %aspectsNNSdobjmanipulatingj theseDTdetaspectsi impactNNpobjonh howeverRBadvmodexisting g %biomechanicsNNSconjanatomyf anatomyNNpobjfrome appliedVBNamodanatomyd fromINpreptheoryc theoryNNpobjonb impliedVBNamodtheorya basedVBNconjneeded` generallyRBadvmodbased_ areVBPauxpassbased ^ #patternsNNSconjpositioning ] movementNNcompoundpatterns\ #positioningNNpobjfor[ #bodyNNcompoundpositioningZ !guidelinesNNSdobjenhancesY enhancesVBZccompdetermineX furtherRBadvmodenhancesW phaseNNpobjofV !concentricJJconjeccentricU orCCcceccentricT eccentricJJamodphaseS eitherCCdetphase R %durationNNdobjmanipulating"Q %manipulatingVBGcsubjenhancesP %whetherINmarkmanipulatingO determineVBadvclneededN neededVBNconjrecommendM isVBZauxpassneededL researchNNnsubjpassneededK moreJJRamodresearchJ sVBZconjbetweenI  _SPdepbetweenH betweenINprepemployingG employingVBGadvclneededF alsoRBadvmodemployingE whileINmarkemployingD lengthsNNSpobjatC longJJamodlengthsB atINpreptrainingA !emphasizesVBZrelclrom@ aDTdetrom? employVBccomprecommend> shouldMDauxemploy= onePRPnsubjemploy< thatINmarkemploy; recommendVBPadvclaims: wePRPnsubjrecommend9 romXXdeprecommend8 motionNNpobjof7 rangeNNconjtempo6 andCCcctempo5 tempoNNpobjas4 !repetitionNNcompoundtempo3 typeNNcompoundtempo2 #contractionNNcompoundtype&1 !#kinematicsNNScompoundcontraction,0 -#exercisespecificNNPcompoundcontraction/ asINprepvariables. suchJJamodas- variablesNNSpobjon , !focusingVBGadvclmaximizing+ !maximizingVBGpcompfor* forINpreptechnique) properJJamodtechnique( #constitutesVBZpcompon' #whatWPnsubjconstitutes& onINprepevidence% !evidenceNNdobjsynthesize$ existingVBGamodevidence# !synthesizeVBxcompaims" aimsVBZconjshown! reviewNNnsubjaims  narrativeJJamodreview thisDTdetreview #hypertrophyNNdobjmaximize #egNNcompoundhypertrophy' ##adaptationsNNScompoundhypertrophy maximizeVBxcomptrying tryingVBGadvclaims !especiallyRBadvmodwhen techniqueNNpobjof! exerciseNNcompoundtechnique  #appropriateJJamodtechnique !ofINprepimportance !importanceNNdobjemphasize !theDTdetimportance emphasizeVBPadvclpromote oftenRBadvmodemphasize 'practitionersNNSpobjin engagingVBGcsubjemphasize whenWRBadvmodemphasize sizeNNpobjin muscleNNcompoundsize inINprepincreases
 increasesNNSdobjpromote	 promoteVBxcompshown toTOauxpromote shownVBNROOTshown beenVBNauxpassshown hasVBZauxshown rtNNPnsubjpassshown trainingNNnsubjpassshown" !resistanceNNcompoundtraining  !!regimentedJJamodresistance   � ����bE,������_@"
����y_C'�����mP8�����`="
�
�
�
�
]
9
	�	�	�	�	z	^	>	!	���w`D+����z^G"����mJ0����vYB!�����nK4�����nP7����pV>�����t]<                  � #defineVBconjhighlighted� !explicitlyRBadvmoddefine� notRBnegdefine� doVBPauxdefine� theyPRPnsubjdefine�  mentionedVBNaclvariables� someDTpobjto�~ referringVBGadvclinferred�} authorsNNSnsubjreferring�| inferredVBNadvcldefine�{ stagesNNSpobjin�z earlyJJamodstages�y learningVBGpcompof#�x ##highlightedVBNROOThighlighted�w #haveVBPauxhighlighted$�v ##researchersNNSnsubjhighlighted�u dependingVBGprepvary!�t +varyVBPrelclrecommendations�s includeVBPconjis�r %prescriptionNNpobjfor�q standsVBZadvclis�p scarceJJacompis�o outcomeNNnsubjis�n exploringVBGadvclis�m directlyRBadvmodexploring�l althoughINmarkexploring�k preventVBxcompenhance�j wellRBadvmodas �i 'effectivenessNNdobjenhance�h emphasisNNattris�g practiceNNpobjin�f !bothCCpreconjliterature�e #prescribingVBGadvclis�d alignmentNNcompoundrom!�c 'orchestrationNNdobjinvolves�b involvesVBZccomppertains�a injuryNNpobjof�` !riskNNdobjminimizing�_ !minimizingVBGadvcltargets�^ targetsVBZccompensure �] #effectivelyRBadvmodtargets�\ ensureVBadvclpertains�[ movementsNNSpobjof�Z bodilyJJamodmovements�Y executionNNpobjto �X !controlledVBNamodexecution�W pertainsVBZROOTpertains�V suggestVBPROOTsuggest�U !standpointNNpobjfrom�T thusRBadvmodmaximize�S powerNNcompoundtraining�R requiringVBGaclgoals"�Q #potentiallyRBadvmodrequiring�P goalsNNSpobjby�O specificJJamodgoals�N !’sPOScaseindividual�M !individualNNpossgoals�L byINagentaffected�K affectedVBNccompnoting�J heavilyRBadvmodaffected�I beVBauxpassaffected�H canMDauxaffected�G notingVBGxcompworth�F worthJJacompis�E positionNNdobjinvolved�D stanceNNcompoundbody&�C -widthorientationNNcompoundstance$�B -gripNNcompoundwidthorientation�A correctJJamodposition�@ involvedVBNaclgroups�? groupsNNSapposvariables�> primaryJJamodmuscle�= followingVBGamodvariables�< includesVBZrelclchecklist�; checklistNNpobjas!�: !describesVBZrelclcomponents�9 manualNNnsubjdescribes�8 nscaNNPcompoundmanual!�7 #associationNNcompoundmanual �6 %conditioningNNconjstrength�5 #strengthNNnmodassociation�4 #nationalJJamodassociation�3 !componentsVBZattris �2 #constituentNNconjtechnique�1 #itsPRP$possconstituent�0 !literatureNNpobjin �/ !!scientificJJamodliterature!�. !!definitionNNnsubjcomponents �- !!agreeduponVBamoddefinition#�, #!universallyRBadvmodagreedupon�+ !noDTdetagreedupon�* thereEXexplis�) currentlyRBadvmodis�( enhanceVBccompproposed�' mayMDauxenhance�& proposedVBNconjshown�% anDTdetexercise�$ upRPprtmake�# makeVBPrelclvariables�" differentJJamodvariables�! describeVBxcompused�  usedVBNacompis� ”''puncttechnique� “``puncttechnique� termNNnmodtechnique� effectiveJJamodrt� componentNNpobjas� keyJJamodcomponent� referredVBDaclvariable� variableNNnsubjpassused� anotherDTdetvariable� failureNNpobjto� muscularJJamodfailure� momentaryJJamodfailure� proximityNNdobjperformed� effortNNpobjof� intensityNNpobjper� weekNNpobjper� groupNNpobjper� perINprepperformed� performedVBNaclsets� setsNNSpobjas� ieFWcompoundsets�
 volumeNNcompoundsets�	 certainJJamodvariables � %manipulationNNdobjrequires� requiresVBZconjshown� knownVBNaclsize$� %%introductionNNROOTintroduction   � ����bC" ����v\=�����jF"����]9�����fP8
�
�
�
�
�
v
W
>
	�	�	�	�	�	i	J	)	
�����tS8����sQ*����lL,����~hD#����tX:�����x]<!����oU2����uS9                � #significantJJamodchanges� wereVBDccompnoted� %additionallyRBadvmodnoted#� !#cautiouslyRBadvmodinterpreted� #interpretedVBNadvclis�  podNNPpobjvia� bodNNPcompoundpod#�~ +plethysmographyNNPcompoundpod �} %displacementNNPcompoundpod �| %airNNPcompounddisplacement�{ viaINprepassessed"�z #compositionNNcompoundchanges�y assessedVBNconjmeasure�x ratherRBadvmodassessed�w changesNNSdobjmeasure�v measureVBccompnote�u noteVBxcompis�t importantJJacompis�s comparedVBNadvclincrease�r lbmNNPdobjincrease�q massNNPcompoundlbm�p leanJJamodmass�o increaseVBccompfound�n !prolongingVBGpcompvs�m !vsINprepconditions�l !conditionsNNSpobjamong�k wasVBDadvclincrease�j nogueiraNNPcompoundal�i keelerNNPcompoundal�h mostlyRBadvmodstem�g stemVBxcompseems�f usingVBGpcompfor�e simplyRBadvmodsuggested!�d )recommendationNNdobjprovide�c protocolsNNSpobjamong�b 'amongINprepheterogeneity!�a 'heterogeneityNNnsubjprovide�` topicNNpobjon�_ givenVBNprepseems�^ bestJJSacompbe�] seemsVBZccompfound�\ !fasterRBRadvmodconcentric�[ !slowerJJRamodrepetition�Z #combinationNNnsubjseems�Y influenceNNdobjreviewed�X reviewedVBDROOTreviewed�W wilkNNPnsubjreviewed�V recentlyRBadvmodwilk"�U %hypothesizedVBNadvclcritical$�T !%previouslyRBadvmodhypothesized�S criticalJJacompbe�R highlightVBccompnoted$�Q +findingsNNSpobjnotwithstanding�P +notwithstandingINpreplead�O outcomesNNSpobjto �N %hypertrophicJJamodoutcomes�M leadVBccompanalyze�L couldMDauxlead�K %combinationsNNSnsubjlead�J analyzeVBccompnoted�I didVBDauxanalyze�H notedVBDadvclfound�G #similarJJamodhypertrophy�F resultedVBNccompfound�E durationsNNSpobjof�D wideJJamodrange�C foundVBDROOTfound�B alNNPpobjby�A etFWcompoundal�@ !schoenfeldFWcompoundal�? %metaanalysisNNconjreview�> !systematicJJamodreview�= temposNNpobjof�< veryRBadvmodslow�; #detrimentalJJamodeffect�: indicatesVBZconjsupported�9 supportedVBNconjproposed�8 poorlyRBadvmodsupported�7 remainVBPauxpasssupported�6 strategyNNpobjof�5 benefitsNNSnsubjremain�4 butCCccactions�3 mannerNNpobjin�2 slowJJamodmanner�1 superRBadvmodslow�0 evenRBadvmodproposed�/ !performingVBGpcompof$�. 'professionalsNNSnsubjemphasize�- commonJJacompis�, orderNNpobjin�+ employedVBNccompproposed�* !eachDTdetrepetition�) duringINprepinclusion�( actionsNNSpobjof�' combinedVBNamodactions�& inclusionNNdobjinvolves�% usuallyRBadvmodinvolves!�$ %conventionalJJamodtechnique!�# !discussingVBGadvclmentioned�" !mainJJamodcomponents�! #provideVBconjconstitutes�  !techniquesNNSdobjtraining� #approachNNdobjconstitutes� focusedVBNrelclstudies!� %specificallyRBadvmodfocused!� #includingVBGprephypertrophy� !developVBadvclsynthesize� !currentJJamodliterature� !endeavoredVBNconjdefine� dateNNpobjto� !paperNNdobjcomprising� scholarlyJJamodpaper� !comprisingVBGaclvariables� variousJJamodvariables� readingVBGpcompby� optimalJJamodtechnique� #extrapolateVBadvcldefine� itselfPRPapposstudy� studyNNpobjof� sectionNNpobjin� methodsNNScompoundsection� !outlinedVBNaclguidelines� examiningVBGaclstudies�
 studiesNNSdobjcreating�	 previousJJamodstudies� similarlyRBadvmodstudies� ambiguityNNdobjcreating� creatingVBGacltechnique
" � ������������ufYKA7(���������udTC4"���������reWI:0��������kTH7&���������ym_TG?5*

�
�
�
�
�
�
�
�
�
�
w
i
Z
K
=
-
!

	�	�	�	�	�	�	�	�	�	}	n	`	U	F	:	,		����������pdVG:. ���������|p`QF=�4) �����������vk`WI>1# ���������rdSB/!����������wl_PH:+��������}nbSK=2&����������xk_PC5" abdominisl  �anthropometryanteriorpanswers7answering	answeranother �
anoth�
ankrde
ankle
ankit�'animalderived	Uanimal�'anglespecific
angles
angle�#anecdotally�5androulakiskorakakis�androgen�andrew�
andor1andersonband6ancillary�ancies	�anchorageanchor#ancetrained
�	ance�anatomyf!anatomical�
aided�analysisanalysing9analyses�analysed	anaerobicx-anabolismrelated�anabolism
anabolic	han �amplitude�amplify

ampletampkα�	ampk`amounts�amount�
amotl�
amongb'aminoacidemia	>
amino	ami	8#ameliorated?ambiguity%amalgamation
alwaysCalthough �'alternatively�#alternative�#alternating;altering�#alterations�!alteration
alter�alt�alsoFals	�alreadySAalphamethylaminoisobutyricoalongside�
along�
alone�almost�allows�allowing�allowed	
allow�allcause~all�
ality�
alike�alignment �aligned	^
align	dalfred�alexander]ale
�albumin1albeit�alabama�alBaktmtor�%aktmammalian�akt�air|aims"
aimed(aim�aid�
aicar�ahtiainen�!agreedupon �agreedo/agonistantagonist�)agonistagonist�agonist�
aging�agents�
agentMage�	agaragainst�
againR
after�)aforementioned
affordedafford�affects_affecting_affected �affect-aerobicendurancewaerobic�	aero	aere�advocated�advisory}advised�advisable�advice^!advantages�%advantageous�advantage	�advancing<advanced{adults3adulthood"
adultpadrenals/adp↔atp�adp�adopting�adoptedu
adopt!admittedly	'%administered_adjusted�adjustadjacent5adhesionstadhesion�adherenceadhere�!adequately�adequateadenosine�adductorsOadductor+
adduc}addressed�address�additive$%additionally�!additionaladdition�adding�
added>addoadaptive�#adaptations!adaptation�
adaptE	adap�ad
Sacutely	�
acuteactually�actual�	acts�+activityrelated;activityactive
�activator�!activation�!activating�activates�activated:activate\actions(action�#actinmyosin�#actinlinked�actingr/actincytoskeleton�/actincrosslinking�
actin�#acteristics�act�across�+acknowledgmentsP-acknowledgements�#acknowledge�-acidsynthesizingv
acids	9acidic�)acidgeneratingC	acid	achieving{achievesachieved�achieve
Caccuracy�%accumulationaccrual accretion5
accre		accounted�account�#accordingly�according}!accordancevaccord	�%accomplished
U%accompanyingaccompanyc#accompanied�!acceptable�#accentuated�!accentuate.acUabundancej+abstractstitlesmabstractsJabstract'abstrackrpabstained8absolute�absent�absence�	abovex
about	�abolishes:	ably
	able	Zablation�ability�abilitiesabducts+abductorsPabdominusj!abdominalsR'abbreviations�ab
ya@   �  
�  � 
b I
 �
\ �% �����qeVN@5)���������{nbSF8%A����������o_N�B5* ��������rcTG?-/�����������zm[MA4"���������whYM=,�����������ugSE;0#	 
�
�
�
�
�
�
�
�
�
p
b
S
D
4
$

	�	�	�	�	�	�	��	�	�	z	n	a	R	:	*		���������vldYOH>.%�R����������{pbTL@3'�������%����ulaTD7)	���������������mVA4$����������i\         	behmc%architecture^#antioxidant�	balli  �brandao�-branchsuggesting9branches@	brad�brachii�-braceimmobilized�bovine	e
bouts�	bout@
bound0	both �
borne�
bones�	bone[)bodyweightmeal	R!bodyweightqbodyfat�+bodycomposition�/bodybuildingstyle�%bodybuilding�%bodybuildersbody[bodily �bodiesibod
boardtbly'blunted"
blunt�!bl%applications�!augmenting�-antiinflammatory�'anthropometryanteriorpanswers7answering	answeranother �
anoth�
ankrde
ankle
ankit�'animalderived	Uanimal�'anglespecific
angles
angle�#anecdotally�5androulakiskorakakis�androgen�andrew�
andor1andersonband6ancillary�ancies	�anchorageanchor#ancetrained
�	ance�anatomyf!anatomical�%betweenstudy�%betweengroupwbetweenHbetter[	best^besides�ber
^	bentbenefits5benefit�!beneficial+beneath�
bench�ben
below�believed	�belief�beings�
being�
begunLbeginsubeginning�
beginZbefore
6befits�beenbed�becoming	�becomes	become�becausePbecame	be �bazvalle�bayesian�
basisbasing
basic!baseline�	basedabaseball'
basal�	bars�barbellqbare
bands=banded�balanced balance3!bagfocusedbag�!background�	backTb�	axis-
axial�ax*	awayavoiding�avoided�
avoidy
avian�average�available %availabilityy
avail	Yauxiliary�'autoregulated�3autophosphorylation3autophagyk!autophaghy�1autocrineparacrineRautocrineauthors �author�australia�augustiaugments�augmented�augment�auckland�
ature!attributes!attributed
3attracts{attract�#attenuating�!attenuatesU!attenuated�attenuate�attention�attempts�!attempting�attempted�attempt�attainedyattaing!attachmentEattached�atrophy�atrophied�!atpbinding�atp�
ative�ationallycational�
ation,	atic�athletic�)athletetrainee,athletesZathlete
�athatetohigh
R	atedate	�atB
asyet!#assumptions�!assumption�assuming�assumed�assume�%associations�'associationalT#association �!associated_associ�assistingHassistant�!assistance�assistassigned�assign�#assessments�!assessment~assessing?assessedyassess	aspirenaspectskaspect�as/%artificially�articleEarthurarteries�%arrangements�aroused�aroundt	arms�arm
:arising"
arise�arguments>argument3argued

argue�arguably�
arguaQ	arguP	area�are_'architecturalq%arandjelovic�ar*aqueous�aquaporinuaptations�aptation�'approximately
�approvedOapproval�'appropriately#appropriate!approaches�approach!appreciate�#appreciablym
appre�applying�
applyapplies4appliede#application�!applicablev'applicability�appears8appearingPappear�apparent�apoptosis�
apart�aphanything\anyDantretter,    ����hP3����tN7
����kI-����sY;���|Z5
�
�
�
�
|
]
;
	�	�	�	�	~	e	D	&	����mXA%����qQ2�����hN/ ����jQ5�����jD+����wT4�����jD-����kQ6                  � kassianoNNPcompoundal� superiorJJacompwas� wolfNNPcompoundal� shortJJamodlengths�  #distalJJamodhypertrophy� elicitingVBGacllengths�~ #regionalJJamodhypertrophy�} suggestsVBZROOTsuggests#�| 'interestinglyRBadvmodsuggests�{ lengthNNpobjof#�z )distinguishingVBGpcompwithout�y withoutINpreprom"�x %dichotomizedVBDadvclindicate �w %papersNNSnsubjdichotomized�v partialJJamodrom�u overINpreprom�t utilizingVBGpcompto�s benefitNNdobjindicate�r indicateVBPrelclreviews �q %metaanalysesNNSconjreviews�p reviewsNNSpobjwith�o !consistentJJacompis �n #recommendedVBNadvcldefined�m achievedVBNrelcldegree�l largestJJSamoddegree�k hereinRBadvmoddefined�j fullJJamodherein#�i 'traditionallyRBadvmodexercise�h jointNNpobjat�g occursVBZrelcldegree"�f #degreeNNnsubjpassrecommended�e definedVBNccompsuggests�d forcesNNSpobjon�c 'gravitationalJJamodforces�b solelyRBadvmodon�a relyingVBGpcompthan�` thanINcccontrols�_ weightNNpobjof�^ descentNNdobjcontrols�] controlsVBZccompensure$�\ %!sufficientlyRBadvmodcontrolled�[ advisableJJacompis�Z willMDauxlead�Y unclearJJacompis�X !acceptableJJamodtempos�W plethoraNNpobjfor�V allowingVBGconjappear�U occurVBccompappear�T appearVBROOTappear�S indeedRBadvmodappear�R doesVBZauxappear�Q #applicationNNpobjfor �P #practicalJJamodapplication�O #conclusionsNNSdobjdraw�N #strongJJamodconclusions�M drawVBaclability�L abilityNNdobjpreclude�K precludeVBconjhighlight �J #uncertaintyNNdobjhighlight �I #conflictingVBGamodfindings�H onlyRBadvmodin�G fiberNNcompoundarea�F iiaJJamodfiber�E iPRPdobjtype�D lateralisNNPcompoundtype �C !exercisesNNSdobjperforming�B allDTdetexercises�A gilliesVBZccompfound�@ lastlyRBadvmodgillies�? !seccentricJJpobjbetween�> areaNNcompoundchanges�= )crosssectionalJJamodarea�< thighNNPnmodarea�; squattingNNpobjduring�: parallelJJamodsquatting�9 effectsNNSdobjexplored�8 exploredVBDconjfound�7 shibataNNPcompoundal#�6 'alternativelyRBadvmodexplored�5 !extensionNNdobjperforming�4 legNNcompoundextension�3 trainedVBNamodmen �2 !quadricepsNNcompoundmuscle�1 !marginallyRBadvmodgreater�0 pearsonNNPcompoundal �/ 'contrastinglyRBadvmodfound�. thicknessNNpobjin"�- medialisNNPcompoundthickness�, vastusNNPcompoundmedialis�+ #experiencedVBDccompfound�* #longerJJRamodrepetitions�) #overallJJamodhypertrophy�( #limbNNcompoundhypertrophy�' lowerJJRamodlimb�& #repetitionsNNSpobjduring�% lookedVBDccompfavored�$ recentJJamodstudy�# extendedJJamodphase�" favoredVBDconjshowed�! sizesNNSnsubjfavored�  twoCDnummodgroups� #differencesNNSdobjshowed&� '#statisticallyRBadvmodsignificant� showedVBDccompnoted� #absoluteJJamodhypertrophy � #elicitedVBNamodhypertrophy� pereiraNNPcompoundal� responseNNdobjenhance� rtinducedJJamodresponse� extendingVBGcsubjenhance� menNNSpobjof� olderJJRamodmen� cohortNNpobjin#� #femorisNNPcompoundhypertrophy� rectusNNPcompoundfemoris� brachiiNNPpobjin� bicepsNNScompoundbrachii� greaterJJRamodbrachii"� 'significantlyRBadvmodgreater� %demonstratedVBDconjnoted� contrastNNcompoundal� absenceNNpobjin�
 observedVBNccompstating�	 capacityNNconjstrength� aerobicJJamodcapacity� statingVBGaclgroups� pretopostNNamodchanges   | ����mV<����y]:����^8����{bC&����iP.
�
�
�
�
�
i
O
*	�	�	�	�	�	_	=	$	����iL,	����aG-�����hM1�����cF(����dE+����lN3����~\:����aE+                  �  kneesNNSconjhips� hipsNNSdobjextend�~ extendVBccompensure$�} %!successfullyRBadvmodcompleting�| !completingVBGpcompon�{ 'jointspecificNNconjmuscle�z fallingVBGxcompavoid�y !avoidVBadvclperforming�x midfootNNconjheels�w heelsNNSpobjbetween�v #distributedVBNccompensure�u #evenlyRBadvmoddistributed�t %likeINprepinstructions�s correctlyRBadvmodinclude�r squatJJamodexercise�q !barbellNNdobjperforming�p %descriptionsNNSpobjof�o listNNdobjprovides�n detailedJJamodlist�m providesVBZconjintend�l exampleNNpobjfor�k targetNNcompoundmuscles�j safeJJconjeffective�i efficientJJamodeffective&�h +biomechanicallyRBadvmodefficient�g intendVBPccomphighlight�f placementNNpobjon�e barNNcompoundplacement�d #footNNcompoundpositioning�c widthNNconjalignment �b %instructionsNNSnsubjintend�a !principlesNNSconjconcepts#�` '!biomechanicalJJamodprinciples�_ conceptsNNSpobjof�^ 'extrapolationNNpobjof�] insteadRBadvmodbased�\ !optimizingVBGpcompfor�[ betterJJRacompis�Z 'configurationNNnsubjis�Y findVBrelclexercise�X createdVBNrelclproduct �W majorityNNnsubjpasscreated�V inceptionNNpobjto�U ’POScaseexercises�T backRBadvmoddating�S !datingVBGaclrefinement�R !refinementNNpobjof�Q yearsNNSpobjof�P productNNattrare�O patternNNpobjfrom �N #performanceNNdobjdelineate�M delineateVBxcompmeant�L meantVBNROOTmeant�K othersNNSpobjthan$�J %longerlengthNNcompoundtraining�I #predisposedJJacompbe�H !understandVBadvclneeded�G fewJJamodmuscles(�F -longmusclelengthNNcompoundtraining �E #strongerJJRamodconclusions�D )configurationsNNSpobjon�C !mechanismsNNSpobjvia �B #conceivablyRBadvmodenhance�A degreesNNSdobjstopping�@ stoppingVBGconjbenefit�? perhapsRBadvmodstopping�> addedVBNamodbenefit�= shorterJJRamodlengths�< endNNcompoundrom�; !traversingVBGcsubjpromote�: defaultNNcompoundapproach�9 biasesNNSnsubjbe�8 appearsVBZROOTappears!�7 !publishedVBNccompemphasized�6 !emphasizedVBNccompfavored)�5 +)semimembranosusNNconjsemitendinosus"�4 )semitendinosusNNPpobjbetween�3 headNNconjmaximus�2 compositeJJamodmaximus�1 extensorsNNSpobjof�0 hipNNcompoundextensors�/ sawVBDrelclgroup�. whichWDTnsubjsaw�- machineNNpobjon�, multihipNNcompoundmachine�+ adductorNNcompoundmuscles�* maximusNNPpobjof�) gluteusNNPcompoundmaximus"�( !hamstringsNNScompoundmaximus�' abstractNNpobjas"�& !conferenceNNcompoundabstract�% presentedVBNconjlends�$ partialsNNSpobjto�# comparingVBGaclstudy�" #unpublishedJJamodstudy�! asyetJJnmodstudy�  credenceNNdobjlends� lendsVBZconjshowed � 'lateralJJamodgastrocnemius#� 3shortermusclelengthNNPconjrom� grewVBDccompshowed� 'medialJJamodgastrocnemius#� %longermuscleNNPcompoundlength!� 'shortermuscleNNPcompoundrom$� )!plantarflexionNNdobjperforming#� )ankleNNcompoundplantarflexion� 'gastrocnemiusNNpobjof*� 1longermusclelengthNNPcompoundpartial� !additionalJJamodstudies � #publicationNNnsubjcompared� sinceINmarkcompared� analysisNNpobjin � subgroupNNcompoundanalysis"� %includedVBNadvclinterpreting� threeCDnummodstudies � %resultsNNSdobjinterpreting� %interpretingVBGadvcltaken� takenVBNccompsaid�
 mustMDauxtaken�	 cautionNNnsubjpasstaken� saidVBDrelclbrachii� tricepsNNcompoundbrachii� musclesNNSpobjfor� !sufficientJJacompis
 � 	h	_	W	F	<	0	$			����������nYH8,
������������wk_VD2�����������t^QG9#;0QG����������${odWLB5��, ����������uj[KA5(����������sbUL9, ����������zo`RC3"�����������saVH^7*
�
�
�
�
�
�
�
�
�

t
h
[
N
A
3


	�	�	�	�	�	�	�	�		t�����}qcRE5$����m���~n_N=,���������td~VG6#	������rycapillarywm cbufferx
briefsbreaksT	brilbriefly�centrallyH  fcompartment?  �componentsdistinct8!clavicular/  �compromiseC!comprisingcomprised
�#compression�!compressed�'comprehensive~compre3compound/'compositional#compositionzcomposite2composed�!components �component �
compo�#complicates[#complicatedfcomplexescomplex	�!completion!completing|!completelycompleted�complete'complementary�#competitorsl#competitive�#competition/competing�!compensate�!compelling�'compatibility;%compartments%'compartmental�#comparisons!comparisoncomparing#comparedscompare�'comparatively�#comparative	q!comparable
Dcompany�#communicate�commonly�common-committed@!commercial�
comes�	come�combiningscombined'ccadence�blunting�burnout�buresh{breakdown	brandao�-branchsuggesting9branches@	brad�brachii�-braceimmobilized�bovine	e
bouts�	bout@
bound0	both �
borne�
bones�	bone[)bodyweightmeal	R!bodyweightq
bodyszbodyfat�+bodycomposition�/bodybuildingstyle�%bodybuilding�%bodybuildersbody[bodily �bodiesibod
boardtbly'blunted"
blunt�!bloodbasedG
bloodblocks
Zblocked"blockade�
block
bjsibj_%bisphosphateI!birmingham�biopsy�!biomedical�%biomechanicsg+biomechanicallyh'biomechanical`!biomarkers�biomarker�%biologically5!biological�
bined
�
binds�binding�	bind�binary<big@biceps�
bicep=bicbiases9biased�	bias�#biarticularJbi�bfr
)beyond�
chest>	chen[checks/checklist �checking
chart`'characterizedd%characterizeZ+characteristics�character�charac�	char�/chaperoneassisted�channels'channel#changeswchangedVchange�
changf#challenging	)!challenges	�challenge1chains&
chain,cessationcertainlycertain �cer
m	cept
�	ceps�centuryEcentuated.centric#centrations%cellular�
cellsJ%celldepletedycellbased�	cell7ceivably
�cc0caveats caveat�!cautiously�caution	causing^causes_caused�
causeMcausal�category6#categorized�!categories�catalyzedCcatalyze�catabolic�
casts#castinduced�	cast�
cases	casecascadescascade`	casa�
carryfcarried�
carlo$#caregulated�careero#cardiotoxinfcardio�cardiac�'carbohydrates	;carxcapture	�#capillaries�capacity�capable�!capability�cap
�!candidateshcandidatemcancercanadianYcanada�can �camkiv�camkii�!calmodulin�
cally�calling
�called�	calf�!calculatedcalculate�
calcu�-calciumdependent�calciumca�calcium�#calcineurin�calcaffeine�#cadependent�cadaver�
cable_ca9cX!byproducts�by �butanolrbut4bundles�	bulk
built�buildup�
build\buford�bufferingBbuckner-brought�brooks�
bronx�broadly	`broader%
broad(	brms�
bring�
brils�   �$ ����pT8����nR9!����uX="����v\9����|Z:
�
�
�
�
~
[
A
#
	�	�	�	�	r	X	=	�����oS2����rQ0�����s[A$����mT:���uS4����eF$�����eC$����|bA$                            �  !decreasingVBGpcompby� #deleteriousJJamodeffects�~ possibleJJacompis�} regardNNpobjin�| stateNNpobjgiven�{ stimulusNNdobjimpairing�z impairingVBGadvclresulted�y loadingNNpobjin�x amountsNNSdobjenhance�w excessiveJJamodamounts�v !converselyRBadvmodamounts!�u %advantageousJJamodpositions�t overloadNNnsubjresulted�s loadsNNSpobjof�r heavierJJRamodloads�q setNNpobjin�p amountNNpobjof�o moderateJJamodamount�n concludedVBDconjattempted�m authorNNnsubjconcluded�l beginningNNpobjat�k suppliedVBNadvclchange�j changeVBccompdetermine�i howWRBadvmodmuscular�h raiseNNcompoundexercise�g forceNNconjmomentum�f momentumNNpobjbetween�e externalJJamodmomentum�d %relationshipNNdobjexplore*�c 1%modelingsimulationNNdobjarandjelovic(�b !1mechanicalJJamodmodelingsimulation �a %arandjelovicJJadvclexplore$�` !%indirectlyRBadvmodarandjelovic �_ %albeitINadvmodarandjelovic�^ knowledgeNNpobjto�] additionNNpobjin�\ spinaeNNpobjfrom�[ erectorNNcompoundspinae �Z glutealsNNScompounderector �Y !!assistanceNNdobjpermitting�X !permittingVBGadvclsways�W forthRBconjback�V swaysVBZrelclposture�U mightMDauxinvolve�T driveNNpobjwith�S minimalJJamoddrive�R postureNNdobjinvolve�Q uprightJJamodposture�P involveVBccomprefers�O wouldMDauxinvolve�N useNNdobjcurl�M curlVBPccompallows�L !allowsVBZadvclminimizing�K #stimulationNNdobjdirects�J #maximumJJamodstimulation�I directsVBZrelclapproach�H refersVBZconjplaced�G nonstrictJJconjstrict�F strictJJamodtechnique$�E ##accordinglyRBadvmodcategorized�D #categorizedVBNconjplaced�C isolateVBxcompmeant�B ancillaryJJamodmuscle�A #involvementNNdobjavoiding�@ avoidingVBGpcompof�? placedVBNROOTplaced�> humansNNSpobjin�= livingVBGamodhumans �< %longitudinalJJamodevidence�; cadaverNNconjhuman�: animalNNconjhuman�9 humanJJnmodstudies�8 #mechanisticJJamodevidence�7 painNNconjflexion�6 flexionNNpobjbetween�5 spineNNcompoundflexion�4 lumbarNNcompoundspine�3 %associationsNNSpobjas�2 reductionNNpobjto �1 !commonlyRBadvmodemphasized�0 moreoverRBadvmodcommonly�/ #alterationsNNSpobjof�. potentialJJamodinfluence�- pertainVBPccompnote�, etcFWccattempted%�+ !+pertainingVBGaclrecommendations�* providedVBDconjexamined�) !kineticsNNSconjkinematics�( examinedVBDconjattempted�' #safetyNNconjperformance�& attemptNNpobjin�% exploreVBxcompattempted�$ attemptedVBNROOTattempted�# %questionableJJacompremain �" %implicationsNNSnsubjremain�! predictorNNattris�  goodJJamodpredictor� #necessarilyRBadvmodis� benchNNcompoundpress� pulldownNNPpobjas� latNNPcompoundpulldown� pressNNPcompoundlat� !activationNNpobjon� otherJJamodstudies� pointedVBDccompfound� positionsNNSpobjwith� periodNNpobjover� calfNNcompoundtraining� alteringVBGpcompof� directJJamodstudies� objectiveJJamodresearch� jointsNNSpobjat� kneeNNnmodjoints� armsNNSdobjoptimize� momentNNcompoundarms� !optimizeVBadvclperforming� !deadliftNNdobjperforming� yourPRP$possbody�
 closeRBadvmodkeeping�	 keepingVBGcsubjis� reasoningNNpobjon� outwardJJconjinward� inwardRBadvmodshift� shiftVBccompallow� allowVBccomphighlight� adhereVBxcompavoided� avoidedVBNrelclactions� fullyRBadvmodextend   � ����yW8�����p[; 	����kM1����uU;�����`F.
�
�
�
�
b
K
-
		�	�	�	�	�	r	X	7		����}fG.�����tZ:�����]>����_E������sU@����uX8����nF$����pM3     � companyNNcompoundstrcng� equipmentNNpobjof � fitnessNNcompoundequipment%�  %manufacturerNNnsubjpassemployed� #corporationNNpobjof�~ #tonalJJamodcorporation�} advisoryJJamodboard�| servesVBZROOTserves�{ interestNNpobjof�z conflictsNNSROOTconflicts%�y %availabilityNNcompoundstatement �x consentNNcompoundstatement�w informedVBNamodstatement �v !!applicableJJROOTapplicable�u statementNNROOTstatement�t boardNNcompoundstatement�s 'institutionalJJamodboard�r receivedVBDROOTreceived�q fundingNNROOTfunding�p versionNNpobjto�o #agreedVBDconjcontributed�n readVBNrelclwriting�m !manuscriptNNpobjof�l editingNNconjwriting�k writingNNpobjto#�j ##contributedVBDROOTcontributed�i bjsNNPconjap�h #apNNPnsubjcontributed�g rbNNPcompoundap�f mcNNPcompoundap�e mwNNPcompoundap�d ideaNNpobjof�c conceivedVBNacljn�b pakNNPconjjn�a #jnUHintjcontributed'�` ''contributionsNNSROOTcontributions�_ affectsVBZpcompon�^ #nontargetedJJamodgroups �] %contributionNNnsubjaffects�\ !‘``puncttechniques�[ coachesNNSconjathletes�Z athletesNNSdobjphysique�Y sportNNcompoundathletes�X physiqueVBPrelclconcept�W existVBPrelclpractices!�V #practicesNNSconjperceptions!�U #perceptionsNNSconjtechnique�T conceptNNdobjexplore�S totalJJamodlength�R !preferenceNNpobjon"�Q #emphasizingVBGxcompappearing�P appearingVBGpcompwith�O thoughtVBNadvclflexible�N flexibleJJacompbe�M subjectNNpobjon�L dataNNSpobjof�K paucityNNpobjgiven�J reserveNNpobjin�I !experienceVBccompargued�H assistingVBGamodmuscles�G contraryNNpobjon�F recoveryNNpobjon�E !meaningfulJJamodimpact�D anyDTdetimpact�C !negligibleJJacompbe�B reachingVBGpcompfrom�A farRBadvmodbe�@ unlikelyJJacompare�? tradeoffNNpobjworth�> fatigueNNdobjcurl�= bicepNNpobjduring�< primarilyRBadvmodworking�; workingVBGccompbe�: ifINmarkworking�9 roleNNdobjplay�8 playVBccompis�7 intendedVBNamodgroup�6 takeVBadvcllong�5 unwantedJJamodgroups!�4 #minimizeVBPrelclrepetitions�3 inferiorJJamodresults�2 yieldVBpcompto�1 !unintendedJJamodgroups�0 generatedVBNaclmovement"�/ 'strictlenientJJamodtechnique �. 'lessRBRadvmodstrictlenient�- %intoINprepincorporated#�, %incorporatedVBNccompsuggested�+ withinINprepfit�* fitVBNccompallows�) #technicallyRBadvmodfit�( broadJJamodrange�' tableNNpobjof�& seeVBPcompoundtable�% –:punctof�$ spansVBZrelclduration�# stretchedVBNccompallows�" designedVBNaclprograms�! programsNNSpobjin�  availableJJamodevidence� clarityNNdobjprovide� !facilitateVBadvclrequire� !alterationNNdobjrequire� requireVBadvclremains� #individualsNNSpobjbetween� 'anthropometryNNpobjin� relateVBPadvcladopt� adoptVBxcompadvisable!� #standardsNNSdobjmaintaining� adequateJJamodstandards!� #maintainingVBGadvcloptimize� needsVBZccompremains� remainsVBZconjis� basisNNdobjhas� logicalJJamodbasis� !basingVBGamodguidelines� caseNNattris� trulyRBadvmodis� imposedVBNaclstimulus� affectVBccompargued� !negativelyRBadvmodaffect�
 arguedVBNadvclis�	 assessVBpcompgiven� )circumspectionNNpobjwith� viewedVBNconjattempted� !simulationNNcompoundpaper� resultingVBGadvclbent� rowNNpobjover� bentVBDadvclattempted� spentVBNacltime� !timeNNdobjdecreasing
   �� ���������ujaP<2'����������yl_QG;.#
����������qbSE9 	���������ygZL=3(	���������scRA.	
�
�
�
�
�
�
�
�
w
g
W
G
:
+

		�	�	�	�	�	�	�	�	x	l	Y	I	8	)			���������vbS?+ ��������raUE3) ��������{lYE7(	��������vhUF9(�������{j]M;+�����                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                !constitute%constituents�#constituent �constant
�!consortium%consolidated�consists�!consisting%consistentlyK!consistent�consist�#considering�!considered�)considerations�'considerationD%considerable%consider�consid�)conservativelyl%conservative�%consequently�%consequencesq#consequenceconsentxconsensus�'consecutively�connects�!connective�!connectionTconnect#conjunctionb#confounding�#confounders�!confounded7)conformational:conflictsz#conflicting�conflict�confiningmconfined)configurationsD'configurationZconfident#!confidence�
confi	�!conferring�!conference&confer
�conducted�conductQ!conditionsl%conditioning �condition�
condi�condensedH	cond`%concurrentlyv!concurrentuconcur	#concomitant�%conclusively?!conclusive!#conclusions�!conclusion�concluded�concludebconclu�concerns~)conceptualizedb!conceptual�concepts_conceptT)concentriconly�)concentrically!concentricV)concentrations'concentrationnconcen�conceivedc#conceivablyB#conceivable4conceiva&conceiv
con�computerZ#compromiseds!compromiseC!comprisingcomprised
�#compression�!compressed�'comprehensive~compre3compound/'compositional#compositionzcomposite2composed�1componentsdistinct8!components �component �
compo�#complicates[#complicatedfcomplexescomplex	�!completion!completing|!completelycompleted�complete'complementary�#competitorsl#competitive�#competition/competing�!compensate�!compelling�'compatibility;%compartments%'compartmental�#compartment?#comparisons!comparisoncomparing#comparedscompare�'comparatively�#comparative	q!comparable
Dcompany�#communicate�commonly�common-committed@!commercial�
comes�	come�combiningscombined'combinev%combinationsK#combinationZcombina

combi�comc%colloquiallyzcollege�%collectivelyg!collection�collect�!colleagues
collagen�col�3coimmunoprecipitate8cohort�cohesive�cognizantcofactorsVcoexpress�	cody�	code�%coactivatingW+coachesathletes�coaches[#cndependent�cn�cmjsprint�clusively�
cloudcloser�closelyt
close�climbed�clever�	clescleotide	�clearly�clearer	�clearance�
clear�cle�!clavicular/!classified�classic|clarity
claim�	cjun�	city�citrate�
cited|citationDcising)
cisesn	cise	5)circumspection#circulatingcir�
cific
cientlcialized}ciable�ci�#chronicallychronic�#chromosomal<
chose�1cholesterolderived$choice!   ~ ����`A ����yX5����x[>���mM+����w]9 
�
�
�
�
y
[
=
	�	�	�	�	|	W	5	�����v\E%�����qM6����~bF,����gD*����qN1�����nM1�����fF(����wcI.                   � peakNNconjtime�  testNNcompoundpeak� wingateJJamodpeak�~ mVBPpunctis�} heightNNpobjas#�| +countermovementNNcompoundjump�{ stimulateVBxcompshown�z %consolidatedVBNconjis�y equivocalJJacompis�x promotingVBGpcompfor�w reportedVBNccompargue�v %improvementsNNSpobjfrom#�u %produceVBcompoundimprovements�t promNNPpobjof�s argueVBPconjbecome�r severalJJamodgroups�q desiredVBNadvclargue�p contextsNNSpobjin�o #superiorityNNdobjsuggest�n whilstINmarksuggest�m 'controversialJJamodtopic�l becomeVBNROOTbecome�k sprintNNcompoundtime�j jumpNNcompoundtime�i verticalJJamodjump�h fieldegJJamodtime�g onesNNSpobjto!�f %strongmanNNconjpowerlifting�e %powerliftingNNdobjeg �d %bodybuildingNNdobjrewarded�c rewardedVBNrelclsports$�b #muscularityNNnsubjpassrewarded�a notablyRBadvmodsports�` sportsNNSnmodsports�_ acrossINprepemployed�^ improveVBconjinduce�] induceVBxcompused �\ fullromNNPcompoundtraining!�[ #preventsVBZrelclalternative�Z whereWRBadvmodprevents�Y !personalJJamodpreference�X variationNNpobjfor�W #alternativeNNdobjpresent"�V ##efficaciousJJamodalternative�U presentVBadvclare�T speedNNcompoundsize�S mostJJSamodoutcomes#�R ##manipulatedVBNamodconclusions�Q lowerbodyNNconjupper�P #upperJJamodadaptations�O #clearJJamoddifferences�N testedVBNpcompin�M beingVBGauxpasstested�L aspectNNpobjof!�K #specificityNNcompoundaspect�J existenceNNdobjsuggested�I smallJJpcompto�H favouredVBDcsubjwere�G smdsNNoprdgrouped�F #meanJJamoddifferences$�E %#standardizedVBNamoddifferences�D groupedVBNadvclfavoured�C favourNNpobjin�B ciNNPdobjrevealed�A   _SPdepsmd�@ smdNNcompoundci�? trivialJJamodci�> revealedVBDconjis�= modelNNnsubjrevealed�< analysesNNSconjsubgroup!�; moderatorNNcompoundanalyses�: #exploratoryJJamodsubgroup"�9 !%multilevelJJamodmetaanalysis �8 %bayesianJJamodmetaanalysis�7 extractedVBNadvclfocused�6 %sportsdiscusNNPconjpubmed�5 pubmedNNPpobjof�4 !prismaJJamodguidelines!�3 %registrationNNdobjfollowing�2 %preJJamodregistration�1 bodyfatNNcompoundtype�0 varietyNNpobjon"�/ )systematicallyRBadvmodreview�. #metaanalyseNNPpobjto�- aimNNnsubjwas�, ourPRP$possaim�+ %synthesisingVBGpcompat�* attemptsNNSpobjto!�) #differingVBGamodhypertrophy�( elicitVBxcompused�' growingVBGamodinterest�& !backgroundNNcompoundrange�% collegeNNPpobjof�$ lehmanNNPcompoundcollege�# yorkNNPcompoundlehman�" newNNPcompoundyork�! !cityNNPcompounduniversity�  !ukNNPcompounduniversity&� #!southamptonNNPcompounduniversity%� !!universityNNPcompounduniversity!� !solentNNPcompounduniversity� sciencesNNPSconjhealth� socialNNPamodsciences� !healthNNnmoduniversity� facultyNNROOTfaculty� steeleNNPapposfisher� jNNPcompoundsteele� bradNNPcompoundsteele� fisherNNPROOTfisher� pNNPcompoundfisher� jamesNNPcompoundfisher+� 5androulakiskorakakisNNPcompoundfisher � patroklosNNPcompoundfisher� miloNNPcompoundpatroklos� conflictNNpobjas$� 'construedVBNrelclrelationships� 'relationshipsNNSpobjof� !financialJJconjcommercial#� !'commercialJJamodrelationships�
 conductedVBNccompdeclare�	 declareVBPconjemployed� remainingVBGamodauthors� nippardNNPcompoundfitness� jeffNNPcompoundfitness� oaNNPcompoundfitness� %strcngNNnsubjincorporated
� �	hYI:- 
�
�r
�
�
�
�
�
�
�
w
i
Z_
J
8
)�

	�	�	�	�	�	�	�	{	m	]	L	;	)		��������wfWG7'����������og]M:)@�������P|n[�`TD92, �3��������zlaS?4)$
��N��������viZOD8�0%��$��������������x?�������~m_QB6*���������whYI:*��������xjSF5)secutively�!containing�#customarily�   )concentricallycomputerZ+counterbalancedqcrunchesgcoverageUcortisol#culminating!continuing�y�!containing�#customarily�coupling�cytokine�y cdalities�
dairyO#contributorsdamaged�damage	�dality6cytoplasm9dej	dataLcytokines�create�database�t�
count�couplet#correlationi'damagerelatedusi-damageassociated�
daily<da�d�cγLcytosol%cytoskeleton�%cytoskeletal�)cyclooxygenase�cycling�cycles
�
cycle�cy
�custom 
cusedA
curve
�currently �current
curls�	curl�	cupy�	cuny�!cumulativeX!cumference�cultured/culture�culprit�culminate
�
cular\csv�
csapo�csat
cruit(
crude�
crualV)crosssectional�crosssec�
crosskcriticalScriteriaX	critcredulity�credible:#credibility�credence creatingcreatineFcreatedX!creasinglyhcreases�crease�coverytcovered�
coverr!covariance�course�couraged|counting +countermovement�councilP
couldL!costameresq-costamererelated)costamerebased�3costamereassociated(costamere
costa{'corroborating�#corresponds�'corresponding	�!correspond	�%correlations�!correlated	�correlate�correla�correctlyscorrect �
corre�#corporation
coresY	core�	cord)cor�coplasm�%coordinationicooccurd!convincingj!conversionG!conversely�convergesm#convergence-%conventional$'controversial�controls�#controlling!controlled �controlcontro�contract%contributory�'contributions`%contribution]%contributingQ#contributesA#contributedj!contribute`contribuK'contrastingly�#contrasting�contrast�contraryG/contraindications�'contradictory
%contractionsp#contraction2#contracting	#contractile�continuumL%continuously�!continuousNcontinues,continue�#continuallycontin>contexts�context�contents�!contentionHcontent8contains�containedQcontain�contacted�contact�consume�consult}construed�!constructsconstruct2#constraintsl  �constitutes(!constitute%constituents�#constituent �constant
�!consortium%consolidated�consists�!consisting%consistentlyK!consistent�consist�#considering�!considered�)considerations�'considerationDconsider�consid�)conservativelyl%conservative�%consequently�%consequencesq#consequenceconsentxconsensus�con!culminates�degree�#degradation�deforms�/deformingyapmtorcdeformed�5deformationinitiatedp#deformation�deform�#definitions`!definition �!definitely�definite�defined�define	defiQdefault:deemedr!decrements!decrement�!decreasing decreases�decreased�decreaseqdeclines�decline;declaredFdeclare�%declarations�decision=decipherdeciding-decades{debris�debates�debatedOdebate�dearthOdeadlifts0deadlift�daysweek�#daystoweeks	�	days
�dayh
daugh�datingS	datedatasheet�)datagenerating�databasesg9damageinjuryassociatede   z � ����`@!�����pR/����fC(����z\=�����dI,
�
�
�
�
m
L
0
	�	�	�	�	y	\	>		 ����aD*����qX4����hF*����rT2�����`F(���t\>!���b.�����a@ �   �{ %publicationsNNSpobjfor�z listsNNSnsubjpassscreened�y referenceNNcompoundlists�x screenedVBNconjexcluded�w hadVBDauxscreened�v returnedVBDadvclexcluded�u onceRBmarkreturned�t excludedVBNconjused�s !irrelevantJJoprddeemed�r deemedVBDaclstudies1�q ?httpabstrackrcebmbrowneduNNPcompoundstudies3�p ?abstrackrNNPcompoundhttpabstrackrcebmbrownedu�o screeningNNpobjby$�n +fulltextsNNconjabstractstitles)�m +abstractstitlesNNSnsubjpassexamined�l sectionalJJamodarea�k crossVBconjthickness�j stringNNnsubjpassused�i augustNNPpobjto�h searchedVBNROOTsearched#�g databasesNNSnsubjpasssearched#�f 'pubmedmedlineNNPnmoddatabases�e searchNNcompoundstrategy$�d %restrictionsNNSnsubjpassplaced�c measuringVBGconjusing�b varyingVBGamodrom!�a -groupsconditionsNNSpobjwith�` leastRBSadvmodtwo�_ %interventionNNdobjinvolve�^ englishJJamodstudies�] thesesNNSconjstudies�\ masterNNposstheses�[ doctoralJJamodmaster�Z %peerreviewedJJamodstudies�Y fulltextJJamodstudies�X criteriaNNSROOTcriteria �W %originalJJamodregistration�V changedVBNadvclusing�U thoughINmarkchanged�T templateNNpobjof �S prosperoNNcompoundtemplate�R registerNNdobjusing�Q #prospectiveJJamodregister!�P 'internationalJJamodregister�O %httpsosfiojeNNPpobjon �N %osfNNPcompoundhttpsosfioje%�M %frameworkNNcompoundhttpsosfioje �L scienceNNcompoundframework�K openJJamodframework!�J !registeredVBNadvclconducted�I itemsNNSpobjwith�H reportingNNcompounditems�G preferredJJamoditems�F !morphologyNNconjfunction�E articleNNnsubjaims�D !moderatorsNNSpobjon �C #subanalysesNNSdobjincluded�B multitudeNNpobjon�A !usPRPnsubjunderstand�@ totalityNNdobjassessing�? assessingVBGpcompof'�> -metaanalyticallyRBadvmodassessing�= despiteINprepanalysing�< scarcityNNpobjdue�; dueINprepanalysing �: #upperlimbJJamodhypertrophy�9 analysingVBGpcompfrom�8 abstainedVBDccompwas!�7 #lowerlimbNNPamodhypertrophy!�6 !#functionalJJamodperformance�5 pallaresNNSpobjby�4 #musculatureNNpobjin�3 grgicNNpobjby �2 functionNNcompoundoutcomes�1 andorNNnmodoutcomes�0 'morphologicalJJamodandor �/ !summarizeVBadvclaccentuate�. !accentuateVBconjbe!�- +musculoskeletalJJamodinjury�, )athletetraineeNNnsubjhas�+ !beneficialJJacompbe�* finallyRBadvmodtraining�) rangesNNSdobjinvolve�( pitchingNNpobjlike �' baseballNNcompoundpitching�& tasksNNSnsubjbenefit�% whereasINmarkbenefit�$ moresoNNdobjbenefit�# scrummingVBGrelclexample�" rugbyNNpobjin&�! )sportsmovementNNcompoundpatterns�  fitsallNNPnmodapproach� onesizeNNPnmodfitsall� inducingVBGpcompin#� %taskspecificNNPcompoundmanner� anglesNNSdobjtraining� throughINprepcompared� timesNNSconjjumps� jumpsNNSpobjas� plausibleJJacompis � #divergentJJamodadaptations� responsesNNSpobjin� acuteJJamodresponses!� +mechanisticallyRBadvmodfrom%� %'accumulationNNconjdeoxygenation#� %lactateNNcompoundaccumulation� bloodNNcompoundlactate � 'deoxygenationNNdobjpromote� dynamicJJamodtraining� applyVBPccompis� versaNNconjlengths� viceNNcompoundversa� likelyJJadvmodresults�
 'anglespecificNNattrbe#�	 'isometricallyRBadvmodtraining� impactsNNSnmodevidence� isometricJJamodtraining� +transferabilityNNpobjin� magnitudeNNpobjin#� %meaningfullyRBadvmoddifferent� plausiblyRBadvmodlead� #inherentJJamoddifferences   y ���|^E'����zW( ���tVB$����iO6����vM/
�
�
�
�
�
j
N
-
	�	�	�	�	o	G	&	����y_>�����jI.	���yX;#���iM2�����pF!�����_A#	����fC*���hJ$                   �t fatJJamodpower#�s #subcategoryNNcompoundstrength�r journalNNPROOTjournal%�q +characteristicsNNSconjprotocols*�p #+participantNNcompoundcharacteristics%�o !)predictorsNNSconjmetaregression#�n )metaregressionNNdobjconducted�m accountVBadvclweighted�l varianceNNpobjby �k samplingNNcompoundvariance�j inverseNNamodvariance�i randomJJamodeffects�h !intraJJconjinterstudy�g !interstudyJJamodgroups)�f %%mixedeffectsNNScompoundmetaanalyses�e structureNNpobjas�d nestedJJamodstructure�c calculateVBxcompopted%�b 7sessionsexercisessetsNNSpobjfor�a !multipleJJamodconditions�` patchworkNNconjggplot�_ ggplotNNdobjusing�^ madeVBNccompusing$�] )visualizationsNNSnsubjpassmade"�\ 3httpswwwrprojectorgNNPpobjin'�[ 3teamNNcompoundhttpswwwrprojectorg�Z coreNNcompoundteam�Y rNNPnmodv�X emmeansNNSconjtidybayes�W tidybayesNNdobjusing�V #drawsVBZdobjinterpreted�U posteriorNNPpobjwith�T packageNNdobjusing�S brmsNNSposspackage�R contextNNpobjwithin#�Q %#plausibilityNNdobjconsidering$�P #%consideringVBGconjcontinuously*�O /#probabilisticallyRBadvmodconsidering%�N %#continuouslyRBadvmodinterpreted�M themPRPpobjupon�L uponINprepbased�K alongINprepconducted�J precisionNNconjestimates�I estimatesNNSpobjfor#�H +estimationbasedJJamodapproach�G +httpsosfiofmvrwNNPpobjin*�F +materialsNNPScompoundhttpsosfiofmvrw"�E 'supplementaryJJamodmaterials�D utilizedVBNaclcode�C codeNNnsubjpasspresented�B fileNNROOTfile�A csvVBNcompoundfile(�@ 3transcribedimportedVBNconjobtained�? rohatgiNNPpobjv�> ankitJJamodrohatgi�= -vNNPprepwebplotdigitizer �< -webplotdigitizerNNPpobjvia�; obtainedVBNadvclobtained�: laterRBadvmodsent�9 weeksNNSnpadvmodlater�8 sentVBNrelclrequest�7 emailNNnsubjpasssent�6 secondJJamodemail�5 initialJJamodrequest�4 obtainVBadvclcontacted�3 workNNnsubjpassperformed%�2 #institutionNNnsubjpasscontacted�1 #unavailableJJacompwas�0 #informationNNnsubjwas"�/ #contactNNcompoundinformation�. missingVBGamoddata�- requestVBadvclcontacted�, contactedVBNconjextracted�+ favouringVBGpcomptowards�* rmNNPnsubjfavouring�) towardsINprepbiased�( biasedVBNccompsay�' sayVBxcompis#�& !categoriesNNScompoundoutcomes�% !optedVBDrelclextraction�$ measuredVBNpcompon�# afterINROOTafter�" groupingsNNSdobjnoted&�! +#preregistrationNNdobjmanipulated�  besidesINprepperformed� 'interventionsNNSpobjof"� 'auxiliaryJJamodinterventions� presenceNNpobjof"� )modalityNNoprdextractedcoded� loadNNcompoundmodality� weeklyJJamodsets� numberNNnsubjmean� frequencyNNpobjwith� fullpromNNPnmodfrequency� !proportionNNpobjby"� )groupconditionNNPcompoundrom� !populationNNpobjof � !statusNNcompoundpopulation� %participantsNNSpobjof� sexNNpobjby� ageNNnpadvmodweighted� weightedVBNamodmean� designNNcompoundweighted)� ))extractedcodedVBNROOTextractedcoded � !!extractionNNROOTextraction%� 3httpswwwgradeproorgNNPdobjusing,�
 3gradeproNNPcompoundhttpswwwgradeproorg �	 #communicateVBxcompproduced � #clearlyRBadvmodcommunicate� producedVBNconjis� gradeNNcompoundtable� relatingVBGaclitems� composedVBNconjshown� reliableJJacompbe� pedroNNPcompoundscale� scaleNNdobjusing�  testexNNcompoundscale� metVBDrelclstudies �~ !!assessmentNNROOTassessment!�} !qualityNNcompoundassessment�| %citedVBDrelclpublications   � ����nT1�����qS0����y`I1����}_;����y_E'

�
�
�
�
u
U
:
!
	�	�	�	�	k	L	+	����bF)	����oS;!
���Y8����pM*����sT:%����cG.�����b: ����}`J7                     �t csaNNPcompoundiso�s uNNPnmod°�r romsNNSpobjin�q !respectiveJJamodroms�p improvedVBDROOTimproved�o torqueNNpobjfor�n ticipantNNROOTticipant�m parNNpobjbetween�l pantNNROOTpant�k particiNNpobjwithin�j kuboNNPcompoundal%�i %%martinezcavaNNPROOTmartinezcava%�h %%esmaeeldokhtNNPROOTesmaeeldokht�g gotoNNPROOTgoto�f valamatosNNPROOTvalamatos�e vertNNcompoundtest�d °NNROOT°�c lNNPROOTl�b  
_SPdepal�a rheaFWROOTrhea�` chartNNROOTchart�_ flowNNcompoundchart,�^ AukpcpartpickercomlistcvxrtNNPconjbuild0�] AhttpsNNPcompoundukpcpartpickercomlistcvxrt�\ buildVBROOTbuild�[ runVBxcompused�Z computerNNpobjon�Y coresNNSpobjof�X cNNPdepbuild�W summaryJJamoddata�V processNNcompoundtable�U detailsNNSdobjincluded�T figureNNcompounddetails�S alreadyRBadvmodincluded�R sameJJamoddata�Q containedVBDadvclexcluded�P becauseINmarkcontained �O !eventuallyRBadvmodexcluded �N !ultimatelyRBadvmodincluded �M #eligibilityNNdobjdetermine�L soughtVBNconjscreened �K versionsNNSnsubjpasssought�J abstractsNNSconjtitles �I titlesNNSnsubjpassscreened�H !remainedVBDconjidentified�G removedVBNadvclsearching#�F !duplicatesNNSnsubjpassremoved!�E !searchingVBGadvclidentified!�D citationNNcompoundsearching�C websitesNNSpobjthrough!�B !!identifiedVBDROOTidentified�A levelNNpobjfor�@ parameterNNpobjof�? valueNNdobjgave�> probableJJamodvalue�= gaveVBDrelclintervals�< estimateNNpobjfor"�; 'compatibilityNNnmodintervals�: credibleJJamodintervals�9 intervalsNNSconjmean�8 quantileJJamodintervals�7 thenRBadvmodplotting�6 plottingVBGpcompfor�5 functionsNNSdobjconstruct �4 densityNNcompoundfunctions$�3 #probabilityNNcompoundfunctions�2 constructVBadvcltaken�1 'distributionsNNSpobjfrom�0 validityNNdobjexamine�/ #checksNNSconjconvergence�. !predictiveJJamodchecks�- #convergenceNNdobjexamine �, #chainNNcompoundconvergence�+ examineVBxcompused�* plotsNNSpobjwith�) traceVBPcompoundplots�( !iterationsNNSconjwarmup�' warmupNNpobjwith�& chainsNNSdobjusing�% markovNNPcompoundchains�$ carloNNPcompoundchains�# monteNNPcompoundchains�" estimatedVBNconjincluded�! #likelihoodsNNSpobjin�  countingNNpobjof� doubleJJamodcounting� !constituteVBadvclused� informVBxcompused� priorsNNSdobjused� !uninformedJJamodpriors� #comparisonsNNSpobjfor"� #controlNNcompoundcomparisons� prepostNNcompoundcontrol� designsNNSpobjgiven � 'appropriatelyRBadvmodgiven!� !!calculatedVBNROOTcalculated� lnrrNNPcompoundmodels� outputNNconjfigures� figuresNNSdobjsee� folderNNnmodfigures� filesNNSpobjin$� !!percentageNNcompounddifference!� 'exponentiatedVBNrelclmodels� rrPRPapposmeans� meansNNSpobjof� ratioNNdobjusing�
 logNNcompoundresponse�	 hereRBadvmodpresented� gNNcompoundsizes� hedgesVBZadvclproduced� !differenceNNdobjusing"� %!standardisedJJamoddifference � modelsNNSnsubjpassproduced� sitesNNSadvclexplored� proximalJJamodsites� underINprepused�  !relatedVBNxcompconsidered� !consideredVBNaclheight�~ wayNNpobjin�} measuresNNSnsubjwere�| machinesNNSpobjof �{ weightsNNScompoundmachines�z freeJJamodweights�y aloneJJacloutcomes�x middleNNrelcllength�w conditionNNpobjin"�v /withinparticipantJJamodupper�u proxiesNNSconjpower
4 � ���ymJ^M�?1 �������ueVA�1#�	��������d�����yiXH6 ����������rdUF3'y
�
�
�
�
�
�
�
�
�
|
p
b
TL
A
4
%

�	�	�	�	�	�	�	�	�;L	�	y	l	_	U	H	=	0�	'		�,	�����������x�ki^QC7)�������^������vmbWK�<:W.#?��������qWI?4)������������th_R>0#����������ypbZOD7*!�������q����se�Y+1&��ff 	flux�+fiberassociatedl fre#facilitated�
extra�fibrous�expands~footballj#extensivelyhfairlycfinlandVexplainsB)experimentallyA'falsepositive>feasible<+exercisetrained9� >forming]formative	formance�	form �foreca
fifth}exceptioneforces�forcepro�+forcegenerating'forced�
force�	forc#for*	footdfonsecaufollowup�follows
�following �followedtfollowsfolder	fold	fol�focusing,focused
focus�
focalsfo@flywheel'fly?
fluid�%fluctuations!fluctuates		fluc	flow_	flnc�
flict flexorsNflexor�flexion�flexibleN
flannV
fixed�	five
�fitsall fitomics�)fitnessrelatedKfitness�fit*fisher�firstly#firstinline�
first�	firmD	fink�finite;finished�finger	�#finegrainedMfindingsQfinding*	findYfinancial�finally*
final�
files	file�filamins�5filamincbagdependenti#filamincbagfilaminc�'filaminbagyap Cfilaminbagmtorcyapautophagyfilamin�filamentsfigures)figuredownload�figureTfigrfieldeg�
fieldficult#fibroblasts�fibers
fiber�
fewer�fewGferent fer
�
femur3femoris�female�	feesS
feelsXfeedback�fedstate	fed>
fects	fect�featuresfeaturedOfavouring�favoured�favour�favoring4favored�
favorCfatiguingpfatigue>fat�faster\#fastedstate	~fasted=fashionE	fashL!fascicular�fascicle�farthing�farAfamily	�
false�fallingzfak)failure �
fails
failing�	fail�faculty�factors^factor	fact	�!facilitatefacmfa	eyes�#extremities}'extrapolation^%extrapolated
�#extrapolate!extraction�)extractedcoded�extracted�extract�'extracellular�external�exteriorextent	{extensors1extensor�extensive	/!extensionsSextension�extending�extended�extend~
exten�!expression�expressed	�express
aexpose'exponentiatedexplosive�exploring �explored�explore�#exploratory�#exploration
�!explicitly%explanations�#explanation�explaineduexplain�experts#experimentsG%experimenter�-experimentalists�%experimental�!experiment�#experienced�!experienceIexperi�
exper�expense�expended�expect	�expansion�existsexisting$existence�
existW!exhaustion�exerts�exertion�exerkine[!exercisingQ-exercisespecific0exercises�!exercisers�+exerciserelated�/exerciseregulatedw+exerciseinduced!exercisein	$exercised�1exerciseassociated[exerciseexercis
A	exer	4execution �#exclusively�excludedtexciting#excessively_excessive�except�exceeds
�exceeding�exceed	examples�	fast3)forceproducingfibertypeF!filaminbag
exert� functiofailed^exertingZ!fasttwitchG!familiarly!fibroblast�%facilitating�feeding�fatigued�exposed�   fullproexhibit�)fatigueability�exhibited�firing�!excitation�/extramyofibrillar�expandw   � ����s^D'����}a@	����z[?����xbD'�����~dG.
�
�
�
�
�
j
J
-
	�	�	�	�	�	^	?	�����{K����yX=#����}Z<$����rN+����hR0����lK,����z\@�����iR1     �u accountedVBNROOTaccounted�t nonwarmupNNPcompoundsets�s doneVBNaclsets�r tionsNNSnsubjpassfound�q distribuNNcompoundtions�p dividualJJamodeffect�o regardedVBNccompwas�n purposesNNSpobjfor �m perfectlyRBadvmodcorrelate�l !correlateVBaclassumption�k n’tRBnegcorrelate�j angleNNnsubjcorrelate!�i !#assumptionNNdobjacknowledge�h #acknowledgeVBxcompis�g averageNNnsubjassumed!�f 'probNNcompounddistributions�e alikeINadvmodwas�d senseNNpobjin�c biasNNcompoundanalysis�b individuNNcompoundeffect$�a #terventionsNNScompoundanalysis�` likewiseRBpcompfrom�_ upperbodyNNPnsubjshowed!�^ dalitiesNNScompoundanalysis�] moNNcompounddalities�\ #exclusivelyRBadvmodfree�[ clusivelyRBadvmodrevealed�Z exNNdobjusing�Y #talNNPcompoundhypertrophy�X #disNNPcompoundhypertrophy�W !phyNNcompoundassessment�V hypertroADDapposmuscle�U originNNpobjfrom$�T #assessmentsNNSnsubjpassgrouped �S #assignedVBNconjcategorized!�R )betweensubjectJJamoddesigns�Q limbsNNSpobjfor�P subjectsNNSnsubjused�O displaysVBZccompsuggested�N belowRBadvmoddisplayed�M displayedVBNconjticks�L ticksVBZccomprevealed�K seenVBNROOTseen�J !relativelyRBadvmodfew �I specificsNNSnsubjpassfound�H !regardlessRBadvmoddeemed�G wordsNNSpobjin �F )individualisedVBDccompnote�E #interestingJJacompis�D idealJJacompbe�C methodNNpobjof�B accuracyNNnsubjbe�A personnelNNSpobjby�@ !supervisedVBNconjdefined�? pintoNNPcompoundal�> barsNNSpobjlike�= metallicJJamodbars�< physicalJJamodstops&�; ;goniometerstensiometersVBZpobjas<�: 5;dynamometerselectricNNPcompoundgoniometerstensiometers-�9 !;isokineticJJamodgoniometerstensiometers�8 builtVBNaclstops�7 stopsNNSdobjused�6 tostudyJJnsubjpassdefined�5 variedJJamodrom�4 bLScompoundsmd�3 largeJJacompwere%�2 %%explanationsNNSROOTexplanations�1 intervalNNROOTinterval"�0 !confidenceNNcompoundinterval�/ waistNNpobjwith�. skinfoldJJamodbody!�- %subcutaneousJJamodthickness�, !propulsiveJJamodvelocity�+ rateNNnsubjmean�* maximalJJamodrate�) !unilateralJJamodrate�( halfromNNPcompoundforce�' meterNNcompoundtime�& yardNNcompoundmeter'�% +velocityNNcompoundcountermovement�$ offRPprttake�# standingVBGamodjump�" textNNROOTtext�! altNNcompoundtext�  higherbNNPpobjto� higherJJRadvmodlower� sdVBZadvmodhigher� saNNPROOTsa�    _SPdepnone� noneNNcompoundsa� seriouNNPdepsa� seriousJJROOTserious� ousJJnpadvmodseri� seriNNPROOTseri� trialsNNSdobjdomised� domisedVBNROOTdomised� ranVBDROOTran� fascicleNNcompoundforce� tensionNNcompoundfascicle � #voluntaryJJamodcontraction� relativeNNPcompoundforce� medianNNPcompoundweeks� followupNNPcompoundmedian� digitizedVBNamodrom� depthNNcompounddetails� assumedVBDaclrom�
 rtdNNPcompoundthickness�	 mvicNNPROOTmvic� pedrosaNNPROOTpedrosa� !werkhausenNNPcompoundal� seoNNPROOTseo� !sadacharanNNPcompoundseo� !increasingVBGadvcloutput� vjNNPcompoundheight� cmjsprintNNcompoundtime� wgtNNPcompoundtime�    
_SPdep  
� whaleyNNPROOTwhaley"�~ !torgluteusNNPcompoundmaximus�} adducNNPpobjfrom�| greatestJJSacompwere�{ gainsNNSnsubjwere�z mpvNNPconjrm�y ledVBNROOTled�x rmsNNPcompoundbody'�w %#betweengroupNNcompounddifferences�v metricJJamodstrength�u isoNNPpobjin   � �����u\D����yX7 ���iP2����~fJ*����rU8
�
�
�
�
z
a
G
$
		�	�	�	�	�	]	A	"	����sT8�����dG+	����hK1
�����iG(����{^B(����g> ����wZ9%����mO/     �u explainedVBNROOTexplained�t 'httpsosfiouvdNNPdobjsee�s pallarésFWcompoundal!�r 'overestimatedVBNconjappears�q equalJJacompbe �p %contractionsNNSconjlengths�o #possibilityNNattris�n seekingVBGadvclis!�m !convergesVBZccompconsidered�l ownJJpobjon�k !enoughRBadvmodconvincing�j !convincingJJacompare�i bodiesNNSnsubjare�h dayNNpobjper�g hourNNpobjfor�f #dorsiflexedJJamodposition"�e #maximallyRBadvmoddorsiflexed�d warnekeNNPcompoundal�c 'investigationNNpobjin&�b +#stretchmediatedJJamodhypertrophy�a emergingVBGamodevidence�` !contributeVBccompappear�_ !associatedVBNrelclpathway�^ pathwayNNdobjactivate�] mtorcNNPcompoundpathway�\ activateVBxcompsuggested�[ reachVBxcompbegin�Z beginVBPadvclresult�Y tissuesNNSnsubjbegin�X passiveJJamodtension�W stasinakiNNPcompoundal�V yetRBadvmodcondition�U overheadJJamodcondition�T headsNNSpobjin�S !extensionsNNSnsubjis�R #overheadarmNNnmodelbow�Q !neutralarmJJdobjcomparing�P 'withinsubjectJJamoddesign�O featuredVBDconjis�N flexorsVBZccompsaw�M elbowNNpobjin�L satoNNPcompoundal�K segmentsNNSpobjin$�J #biarticularNNPcompoundsegments�I maeoNNPcompoundal�H mcmahonNNPcompoundal�G sixCDnummodstudies"�F !!reasonablyRBadvmodconsistent�E !oranchukNNdobjoptimising�D !optimisingVBGpcompfor�C !supportingVBGamodevidence�B #substantialJJamodevidence�A wishVBccompsuggests�@ traineesNNSnsubjwish�? goalNNcompoundtrainees�> !hypothesisNNdobjsupported�= decisionNNattris�< binaryJJamoddecision�; proveVBadvclsuggest�: utilisingVBGpobjas#�9 'directionallyRBadvmodfavoured�8 !noteworthyJJacompis�7 largelyRBadvmodtrivial�6 categoryNNnmodmuscle�5 lookingVBGcsubjwere�4 appliesVBZccompsuggest�3 principleNNnsubjapplies�2 stronglyRBadvmodsuggest�1 marginNNpobjto�0 maximiseVBccompappears�/ #competitionNNpobjin �. +httpsosfioahjnfNNPpobjfrom�- rangedVBDnsubjshowed�, pooledVBNadvclappears�+ modestJJamodimpact�* findingNNpobjof�) majorJJamodfinding�( aimedVBNaclarticle �' !!discussionNNROOTdiscussion�& unableJJacompwere�% muchJJdobjobtain�$ !inferencesNNSdobjmake�# !confidentJJamodinferences�" #lackVBPccompreiterating�! manyJJnsubjlack�  #reiteratingVBGxcompworth"� !regressionNNcompoundanalyses� !throughoutINprepmade� secondlyRBadvmodmade� rigorousJJamodprocess� peerNNcompoundreview � undergoneVBNrelclinclusion� resultVBadvclsuffers� firstlyRBadvmodresult� #limitationsNNSpobjfrom� suffersVBZccompprovides� gaugeNNdobjprovides� reasonNNpobjfor� simpleJJoprdkept� keptVBNcsubjprovides� purposelyRBadvmodkept� !relevantJJamodliterature� vastJJamodmajority!� entiretyNNnsubjpassincluded� hopedVBNadvclscreened� checkingNNpobjto)� /referencecitationNNcompoundchecking�
 separateJJamoddatabases�	 analysedVBNrelcltheses)� 1master’sdoctoralNNPcompoundtheses� featuresNNSpobjof� uniqueJJamodfeatures� adherenceNNdobjreporting� pointNNcompoundestimates� completeJJamodreporting� testsNNSpossresults� #statisticalJJamodtests�  programNNpobjacross"� 5titrationprogressionNNpobjat,�~ 5baselineNNcompoundtitrationprogression�} scoresNNSpobjof�| comesVBZccomphad�{ outRBpcompon�z tionNNPpobjby�y condiNNPcompoundtion�x βNNcompoundci�w slopeNNpobjwith�v pactNNattrm
'��D5(��wl^T7D5,"�����������v�h[M>3%������p�������zncY�F�;1�'�����������xma|VI>U5+ �JU�n���������uXND2'`����������|odPC4(����������;��rdVF;1%
�
�
�
�
�
�
�
�
�
�
�
z
i
\
Q
E
:
.��� 
%


	�	�c	�	�	�	��	�	�	�	�	}	j	]	Q	G	;	.C	!	���������'sj_UE3)�
 ������������uiJ?#�������|qfZ�R,���pacting  !highvolumefourth%highervolume 
girth�gradient�-glucosephosphate�glucose�)growthoriented�#freeradical�forecast�c�hydrogen�heighten�
gives�#homeostasis�!generation�� h	form �hydratedlhydrationdhalflivesY
glandTglobulin3helping fusiongh�hepato�hormones�	gata�	fuse�#heighteningxgenderwfullestphwG#fundamental&functions5#functioning	�!functional6function2
fully�fulltextsnfulltextYfullrom�fullprom�	full�fuFfruitful�fromd	frey5!frequently�frequent
�frequency�#frequencies
�frequen
�#freeweights
?!freeweight�freedom�	free�fre
�frameworkM!fractional�	four1founding�
foundCfoster�
forth�	fort	�   8hypertrophied�%hypertrophicN!hypertroph
�'forcevelocity�gastroc�!heightened�heightens�gracilis<
formspforming]formative	formance�hy humans�
human�
hubalJhub�hu3httpswwwrprojectorg�3httpswwwgradeproorg�+httpsosfiowdjxg~'httpsosfiouvdt%httpsosfiojeO+httpsosfiofmvrw�+httpsosfioahjnf.9httpsgtexportalorghome2
https]?httpabstrackrcebmbrowneduqhoweverh
howev�how�
hours	&	hourg
hotlyN	host�horwarth�hormonehormonal
hormo�hormesis
�
hoped	hope�!homodimers�homodimer	hold,%historically!historical�	hips
hippojhip0hindered�'highthreshold))highrepetition7highrephighlyjhighloads
$highload	�!highlights	�#highlighted �highlightR'highintensityhighest
�higherb�higher�	high	�higbie�hickson}'heterogeneityaheteroge

heter�hermeticherein�	here	hensive4!henceforth
hence	�
helps�helpful|helped�	help
helms�	held	�hek�height�
heelswhedges
heavy
heavily �#heavierload�heavier�
heavi�healthy	health�!headtohead	�
headsT	head3heW	have �haustion�	haun�hashardly�	haps"happens happen#	hand�
hance~!hamstrings(hampered7hamilton�hallmark	�halfrom�#halfmaximal�	half|hadwh	x!guidelinesZ
guide
�	gued+'growthrelatedgrowth �growing�	grow-groupsconditionsagroups �groupings�grouped�)groupcondition�
group �grosser{
gross		grip �
grgic3	grewgreatest|greater�
great
}gravity�'gravitational�grantsN
grams	gramGgradepro�graded
o
grade�%gprdependent�	gotog	good�;goniometerstensiometers�	goneAgoldberg�
goals �	goal?go�!glycolytic�!glycolysis�glycogen�gluteus)gluteals�glutamine�global<gli	�gleA
given_	give�giv	�
ginedgillies�
giant�ggplot�
gests�gested�george�genetics
>genetic
�genesis�
genes�generic!generatingxgenerates�generated0generate�generally`generalu	gene	�	gave=
gauge'gastrocnemius	gaps

gapdh�ganize
gains{gaining9gained�	gain7!gadolinium/g
futurezfutile�#furthermore
KfurtherXfundingq   ~ ����oQ8�����qR7�����dH"�����bH+�����d?
�
�
�
�
k
S
8
'
	�	�	�	�	z	^	>		���qO0����sQ3����mP5����vZ;%�����a?"����uV;�����^=����{aD                          #�s !mcmasterNNPcompounduniversity�r #kinesiologyNNPpobjof�q bagNNapposfaculty�p privateJJamodbag�o !technologyNNPpobjof#�n !aucklandNNPcompounduniversity!�m 'environmentalNNPamodscience�l sprinzNNPcompoundfaculty�k zealandNNPcompoundfaculty�j alabamaNNPnpadvmodschool!�i !birminghamNNPnpadvmodschool�h llcNNPpobjfor�g fitomicsNNPcompoundllc�f australiaNNPcompoundllc�e vicNNPcompoundaustralia#�d melbourneNNPcompoundaustralia�c victoriaNNPconjhealth�b instituteNNPpobjof�a schoolNNPnmodinstitute�` usaNNPcompoundschool�_ nyNNPcompoundusa�^ bronxNNPcompoundusa�] cunyNNPcompoundschool�\ !departmentNNPnmodschool�[ vigotskyNNPapposhaun�Z dNNPcompoundvigotsky�Y andrewNNPcompoundvigotsky!�X phillipsNNPcompoundvigotsky�W stuartNNPnpadvmodhaun�V haunNNPROOThaun�U tNNPcompoundhaun�T codyNNPcompoundhaun�S jozoNNPcompoundhaun�R theiuscaNNpobjof�Q standNNpobjin�P athleticJJamodposition�O contentsNNSpobjwith�N interestsNNSROOTinterests�M competingVBGamodinterests�L !finishedJJamodmanuscript�K feedbackNNdobjproviding�J usefulJJamodfeedback�I providingVBGpcompfor�H helmsNNSconjwright�G ericNNPcompoundhelms�F wrightNNPpobjto�E thanksNNSROOTthanks-�D --acknowledgementsNNSROOTacknowledgements�C 'periodizationNNpobjby(�B #'renaissanceNNcompoundperiodization�A projectNNpobjfor�@ phdNNPcompoundproject�? %investigatorNNpossproject�> outputsNNSpobjfor �= picturesNNScompoundfigures�< datasheetNNnsubjfigures!�; scriptsNNScompounddatasheet�: materialNNconjdata�9 #participateVBaclapproval�8 approvalNNROOTapproval�7 ethicsNNScompoundapproval%�6 %%declarationsNNSROOTdeclarations'�5 ''abbreviationsNNSROOTabbreviations�4 effortsNNSnsubjpassmade�3 failingVBGxcompextract�2 extractVBxcompeasier�1 easierJJRconjavailable�0 openlyRBadvmodavailable�/ #replicationNNpobjof�. easeNNpobjfor�- compareVBPccompseeing�, seeingVBGpcompin�+ !interestedJJacompbe�* replicateVBccompapplies!�) +coachesathletesNNSnsubjwish�( soINcceg�' exactlyRBadvmodhow�& typesNNSpobjfor�% %outperformedVBNamodprom#�$ %#recreationalJJconjcompetitive�# #competitiveJJamodcoaches�" whoseWP$possexistence'�! /recognizeVBrelclcontraindications"�  %practitionerNNnsubjrecognize"� !%managementNNconjavailability!� /contraindicationsNNSattrare� adoptingVBGxcompworth� stillRBadvmodis� uncertainJJconjsmall� downsideNNdobjhas� littleJJdobjhas� errorsNNSdobjminimize� negativeJJamoderrors� falseJJamoderrors� viewpointNNnsubjaims� consensusNNpobjof� sortNNpobjto� comeVBxcomprequired� lackingVBGadvclbeing� infancyNNpobjin � %viewingVBGconjunderpowered#� +inconsequentialJJamodvariable� regardingVBGpcompin � !consistsVBZccompconceptual� !conceptualJJacompis�
 #befitsVBZamodresearchers�	 firstJJamodbefits� !viewpointsNNSdobjadopt� readerNNnsubjadopt� drawingVBGxcompadvised� advisedVBNadvclare� %underpoweredJJacompare� #credibilityNNdobjgive� !conclusionNNdativegive� giveVBxcomprequired�  requiredVBNccompsee� rrsNNpobjfor �~ +httpsosfiowdjxgRBadvmodsee�} testingVBGpcompin�| helpfulJJacompbe�{ supportVBconjsurmised�z subNNcompoundgroup�y )musclespecificNNPattrbe�x surmisedVBDconjconcluded�w stemsVBZconjwere�v theirsPRPnsubjstems   �% ���oS(�����rX8����{T;&
�����cJ4�����aA%
�
�
�
�
h
O
:
!
	�	�	�	�	q	U	<	 �����wW7�����uT8�����iH&����|\G%����oU7�����gI$����nF,�����\A%                             �s briefJJamodoverview�r coverVBconjinvolve#�q !bodyweightNNPcompoundexercise�p formsNNSpobjamong�o distinctJJamodforms�n torsNNSpobjamong�m facNNcompoundtors�l #constraintsNNSnsubjaffect�k pendingVBGadvclinvolve�j deINadvmodinvolve%�i )unraveledVBNadvclconceptualized!�h !#structuralJJamodadaptations�g )esNNPconjconceptualized�f changNNPcompoundes�e molecularJJamodes�d 'characterizedVBNamodes�c comNNcompoundposition"�b )conceptualizedVBNconjdenotes�a positiveJJamodchanges�` tissueNNcompoundgrowth�_ sitionNNcompoundstand�^ poNNcompoundsition�] purposeNNpobjfor�\ denotesVBZconjmade �[ #differentlyRBadvmoddefined�Z trophyNNnsubjpassdefined�Y hyperNNcompoundtrophy�X scopicJJamodscale�W microNNamodscale�V #macroscopicNNpobjfrom�U !urementJJamodtechniques�T !measJJamodtechniques�S detectedVBNrelclaccretion�R !measurableJJamodchanges�Q %leadsVBZrelclupregulation�P mpsNNPpobjin�O synthesisNNPcompoundmps�N teinNNPcompoundsynthesis�M proNNPcompoundtein�L %upregulationNNpobjin �K %eventualJJamodupregulation�J cellsNNSpobjby�I stressNNpobjof�H !metabolicJJconjmechanical�G tificNNPcompoundresearch�F scienNNPcompoundtific�E centuryNNpobjover�D pastJJamodcentury�C physioJJamodprocess�B 'understandingVBGpcompin�A progressNNnsubjpassmade�@ !tremendousJJamodprogress�? #onciliationNNpobjof�> #recNNcompoundonciliation�= idenceNNnmodsections�< evNNPnmodsections�; losingVBGconjning�: ningNNpobjbetween�9 winNNcompoundning�8 contentNNconjaccretion�7 cellNNcompoundcontent�6 satelliteNNcompoundcell�5 accretionNNdobjfavoring�4 favoringVBGaclbalance�3 balanceNNpobjin�2 netJJamodbalance�1 shiftsNNSdobjinclud�0 includVBZccompsee!�/ !processesNNSnsubjcontribute�. #hyperplasiaNNpobjin�- istingVBGpcompof�, preexNNPnsubjisting"�+ )ialNNPcompoundcrosssectional�* axJJamodial!�) #energyNNcompoundhypertrophy�( intakeNNpobjby�' proteinNNcompoundintake�& dietaryJJamodintake�% mainlyRBadvmodby�$ enRBadvmodis�# drivJJaclprocess�" adulthoodNNcompoundmuscle�! longtermJJamodevidence�  )reconciliationNNpobjof"� !#constructsNNSdobjelucidating� #elucidatingVBGpcompto� devotedJJacompare� sectionsNNSnsubjare� cificNNcompoundsections� speNNPcompoundsections� !consortiumNNpobjof� representVBPconjdefined� !leticJJamodpopulation� !athJJamodpopulation� skeletalJJamodmuscle� #synthesizesVBZconjdefined� fieldNNpobjin� expertsNNSpobjof� leadingVBGamodexperts� ingNNconjwinning� losNNPcompounding� winningVBGpobjbetween� casesNNSpobjin� !predicatedVBNaclplacings� placingsNNSpobjwith�
 partNNpobjin�	 judgedVBNadvclis$� %bodybuildersNNSnsubjpassjudged� !attributesNNSattrare� !desirableJJamodattributes� 'strengthpowerNNpobjin!� %accompanyingVBGamodincrease� bulkNNnsubjare"� #fibersNNScompoundhypertrophy� #preexistingJJamodfibers�  wholeJJamodmuscle� axialJJamodarea�~ creaseNNpobjas"�} 'operationallyRBadvmoddefined�| ilNNPcompoundusa�{ evanstonNNPcompoundusa'�z %!northwesternNNPcompounduniversity(�y !%statisticsNNPScompoundnorthwestern�x #engineeringNNpobjof!�w !#biomedicalJJamodengineering#�v ##departmentsNNPROOTdepartments"�u #canadaNNPcompounddepartments�t hamiltonNNPcompoundcanada
$ �����r\K:,���������hO7; �������bN��xpfZN?6,!
������d���}t�����d�\SK?6)��f�������~r��� �{jXF������p�TcUC4*#�����.������ubP;&�{������uiXN@1"�
�
�
�
�
�
�
�
�
{
n�
`�
R
E
5/
*

		�	�4	�	�	��o	�	�	�	�	|	n	W	N	@	1			���u� ���>	�����z%m_SE8)������pcYMC8.$������  interactsinhibited�initiate� justhypertro�#hypertophic�#hyperplasia.-hyperinsulinemia	Chyperemia�1hyperaminoacidemia	
hyperYhyldahlI!hydrolyzed�inactive6#inclination2!indicating	#impractical�!inadequate�inorganic�impinging�infection�!initiation�)hypoxicinduced�hypoxic�influxw'incorporationO!inhibiting>!Ehypothalamicpituitarygonadal,igfiea	igfs
igfec
igfeb
igfea
igfbp
igfbps	!inhibitory�!implicated�incline�3hypertrophyspecific|!impressivefidentify2impairimpactsimpactingimpacted�
impactiimmune�)immobilization�#immediately�imental�	imal2imaging
t#illustrated!illustrate0illness�ilkilarlyzil�iia�ii
&!igfinduced4igf�igating
�if:ie �#identifying�!identifies
j!identifiedB)identification%identifiableDidenced@idence=	idencideally�
ideal�	ideadicsbicehockey�icated ically	�	icalicv'ibandspanning�ial+i�hypoxia�)hypothetically
�%hypothesizedU#hypothesize!hypothesis>#hypoosmotic�7hypertrophytriggering 9hypertrophystimulatinga1hypertrophysensing/3hypertrophyoriented/hypertrophyorient
)hypertrophyori5hypertrophymediating~)hypertrophyingc3hypertrophyinducing�7hypertrophyassociated�#hypertrophy'institutionals#institution�institute�instead]instance	winsist�insights
insight�inseriesl!innervatediinnate	�injury �injuries!initiating
initiatesinitial�inhibit�inherent	ings
~
ingly
�/ingestioninfusion	ingestion	ingested	Fing/infusioningestion	,infusion	1	infu	W%infrequently
�informedw#information�inform!influenced5influenceY
influZinferred �inferior3#inferential�!inferences$
infer	�!infeasible�infancy�inertial�inducinginduces
*induced�induce�#individuals)individualised�!individual �individu�individ
�!indirectly�indirect�!indicators !indicative�indicates:indicated
8indicate�	indi:#independent2indeed�	inde'incrementally
�#incremental
�%increasingly
�!increasing�increases
increased�increaseo'incorporatingE%incorporated,#incorporater+inconsistencies
2+inconsequential�%inconclusive'inconceivable�%incompletelylinclusion&includingincludes �includedinclude �includ0incidence�inceptionVinbetween�ination
�inability[in	imum	�improving�%improvements�improvedpimprove�!impossible�imposed#importantly�importantt!importance
imply�impliedb'hypertrophied�%hypertrophicN!hypertroph
�injuredjinjectionginsertionE%inflammatoryD%implications�%implementing
X#implementedi+implementations�)implementationAimpairing�impaired7
inoki�#inefficientY'hypertrophies�!influences�0 intense�%inscriptions>!inhibitors�� 6integrin�isotonic�� )interstitium�	itgaimbalance\!inhibitionq� interferesimpairs� key �	keinducible-   1keeping�	keep�keelerikassianoinhibitor]    ����w\<�����e>����|];����sS1����u^@#
�
�
�
�
h
M
'
	�	�	�	�	p	R	5	�����cC ����zaI/�����d@"����{[=$�����a?#����pP3����iG"	����oT9                    �r !gestsNNSdobjfacilitate�q sugNNcompoundgests�p transientJJacompbe�o ponentsNNSpobjin �n %constituentsNNSconjcoplasm�m coplasmNNPpobjof�l sarNNPcompoundcoplasm$�k -disproportionateJJamodincrease�j creasesNNSpobjin"�i 'comparativelyRBadvmodlimited�h !manifestVBrelclmyofibrils�g terNNpobjinto�f daughNNPcompoundter�e cycleNNnmodtheory�d expansionNNcompoundcycle�c !compellingJJamodtheory�b jorgensenNNPcompoundal�a chronicJJamodjorgensen�` linesNNSnsubjprovide�_ addingVBGpcompincluding�^ genesisNNconjnumber"�] #myofibrilloNNcompoundgenesis�\ splittingNNconjseries�[ seriesNNpobjin�Z increasedVBNamodnumber�Y rizedVBNccompsuggests�X theoNNnpadvmodrized�W myofibrilNNPamodaccretion�V %myofibrillarNNPnmodmps�U regularJJamodincreases�T #weeksmonthsNNSpobjto&�S ''intramuscularJJamodtriglycerides�R stitutesNNPpobjon"�Q 'triglyceridesNNSconjglycogen�P glycogenNNpobjof�O !substratesNNSdobjtain�N !storedVBNamodsubstrates�M tainVBPconjcontain�L iologicalJJamodfunctions�K physNNSnmodfunctions�J cupyNNPrelclenzymes�I ocNNPnsubjcupy�H ionsNNSconjenzymes�G enzymesNNSdobjcontain!�F ribosomesNNScompoundenzymes�E !organellesNNScompoundeg!�D !reticulumsNNScompoundfibers�C beyondINprepoccupied�B brilsNNPpobjby�A myofiNNPcompoundbrils�@ occupiedVBNccompsuggest�? mentNNpobjof�> environNNamodment�= nentsNNSattrare�< compoJJamodnents�; releaseNNconjsite�: storNNcompoundage�9 calciumNNcompoundstor�8 siteNNattris!�7 %sarcoplasmicJJamodreticulum�6 !productionNNpobjin �5 #recruitmentNNpobjfollowing�4 #neuralJJamodrecruitment�3 !sarcomeresNNSdobjcontain�2 containVBPrelclunits�1 unitsNNSnsubjis�0 tractileNNcompoundunits�/ conJJcompoundtractile�. reticulumNNPoprdcalled�- plasmicJJamodreticulum�, sarcoJJamodreticulum�+ calledVBDaclorganelle�* organelleNNconjulum �) #specializedJJamodorganelle�( ulumNNSdobjconsist�' reticJJamodulum�& 'mitochondrialJJamodulum"�% )consistVBPconjmultinucleated#�$ )multinucleatedJJadvclreferred�# bundlesNNSpobjinto�" !fascicularJJamodbundles�! separatesVBZconjoccupy�  occupyVBROOToccupy� #vasculatureNNconjtissue � !myofibrilsVBZadvclreferred� nellesNNSdobjsuspends� orgaNNcompoundnelles� suspendsVBZrelclmedia� mediaNNSpobjas� aqueousJJamodmedia"� !!sarcoplasmNNPnsubjmyofibrils� cellularJJamodfluid� spaceVBPccompis� outsideJJamodmembranes&� ''extracellularJJconjcompartmental� !sarcolemmaNNconjmembranes� membranesNNSpobjbeneath� beneathINprepized� 'intracellularNNpobjinto� 'izedVBNaclcompartmental� 'compartmentalNNattris� fluidJJattris� etalNNcompoundmuscle� skelNNcompoundmuscle�
 collagenNNcompoundprotein�	 !connectiveJJamodtissue � !sheathedVBNccompappreciate� skeleNNPcompoundmuscle� warrantedVBNamodmuscle� cleNNcompoundstructure� musNNPcompoundcle%� #descriptionNNnsubjpasswarranted$� '!hypertrophiesNNSdobjappreciate� !appreciateVBadvcloccupy�  detailNNpobjin� updatedVBNamodmodel �~ 'comprehensiveJJamodreviews�} consultVBxcompcouraged�| couragedVBNacompare#�{ '!physiologicalJJamoddiscussion�z !nuancedJJamoddiscussion�y readersNNSnsubjare�x modeNNconjture�w tureNNdobjprovides�v naNNcompoundture�u generalNNpobjregarding�t overviewNNdobjprovides   �J �����r\H-�����~bF0�����thP-�����)����ydJ0

�
�
�
�
q
V
E
9
'
	�	�	�	�	�	�	�	p	[	F	:	.	#	�����{k\PC-����ucL;$������wgTA$ ������pXB/�����k�����oT
�����`C)������u^J                    � !myofibril
ORG� !retic ulumPERSON� !sarcoplasmORG� !tal muscleORG
� facORG� chang esPERSON� %micro scopicORG� mpsPERSON&� Kscien tific research mechanicalORG� -the past centuryDATE� #onciliationCARDINAL� 3preex isting muscleORG� ;ial crosssectional areaORG�
 los ingGPE�	 -evanston il usa
ORG� canadaGPE� hamiltonPERSON6� kauckland  new zealand department of kinesiologyORGG� �environmental science auckland university of technology privateORG� /faculty of healthORG4� galabama sports performance research instituteORG� !birminghamGPE� %fitomics llcORG�  )vic  australiaORG melbourneGPE ~ ?sport victoria universityORGM} �social sciences solent university southampton uk institute for healthORG*| Sbronx ny usa school of sport healthORG{ lehmanORG$z Gdepartment of health sciencesORGy )james steele  PERSONx phillipsORG6w ejozo grgic cody t haun eric r helms stuart mPERSONv james pPERSONu !eric helmsPERSONt %james wrightPERSONs firstORDINAL
r rrsORGq +httpsosfiowdjxgORGp grgicPERSONo pallarésNORPn ;upperbody muscle groupsORGm 'httpsosfiouvdORGl six weeksDATEk an hourTIMEj /stasinaki et al  PERSONi !neutralarmORGh !maeo et alORGg satoPERSONf 'mcmahon et alORGe sixCARDINAL	d alORGc -httpsosfioahjnf
PERSONb secondlyORDINALa threeCARDINAL` 7height metaregressionPERSON_ nonwarmupPRODUCT^ )metaregressionORG] dividualGPE\ #terventionsPERSON[ upperbodyORG$Z Gclusively resistance machinesORGY )dis tal musclePERSONX hypertroPERSONW )betweensubjectORG6V edynamometerselectric goniometerstensiometersPERSONU seriou saPERSONT 'halfrom forceORGS seriORG
R romGPEQ weeksDATEP mvicPERSONO pedrosaPERSONN -werkhausen et alPERSONM #ticipant  lPERSONL 1wgt cmjsprint timeORGK %whaley et alORGJ pallaresPERSONI 1torgluteus maximusPERSON	H rmGPE
G rm
GPE
F mpvORGE u  °  °PERSOND l  °  °PERSONC ticipant
PERSONB kuboORGA %martinezcavaPERSON@ gotoPERSON? valamatosPERSON> oneCARDINAL= %metaanalysesORG< !interstudyGPE; 7sessionsexercisessetsNORP: tidybayesGPE
9 csvORG8 ankitPERSON7 /a few weeks laterDATE6 secondORDINAL5 3prom groupconditionORG4 weeklyDATE,3 Qgradepro httpswwwgradeproorg  thisPERSON*2 Sabstrackr httpabstrackrcebmbrowneduORG1 augustDATE0 'pubmedmedlineORG/ %at least oneCARDINAL. %at least twoCARDINAL- englishLANGUAGE-, Ythe international prospective registerORG+ 3metaanalyses prismaORG* %schoenfeld  PERSON) onesizeORG( %recent yearsDATE' bayesianNORP& %sportsdiscusORG[% �3social sciences solent university southampton uk city university of new york lehmanORG$ ;faculty of sport healthORG# %james steelePERSON" bradPERSON&! Eandroulakiskorakakis james pPERSON  3milo wolf patroklosPERSON %jeff nippardPERSON! Ainstitutional review boardORG +pak mw mc rb apORG
 pakORG only oneCARDINAL -nonstrict strictGPE -exercisespecificORG$ Gexercisespecific instructionsORG 3rt exercisespecificORG yearsDATE! Aexercisespecific techniqueORG %longerlengthNORP -longmusclelengthPERSON 3shortermusclelengthORG %longermuscleORG 1longermusclelengthGPE !only threeCARDINAL kassianoPERSON wolfPERSON  ?thigh crosssectional areaGPE twoCARDINAL
 pereiraGPE	 +pereira et al  PERSON mass lbmPERSON !al  keelerPERSON nogueiraPERSON keelerPERSON wilkPERSON !schoenfeldPERSON 7exercisespecific bodyORG" Cexercisespecific kinematicsORG   �a �����t`>-�������lZI*������q[K:������lP;+������uWB-
�
�
�
�
�
�

h
N
A
)
	�	�	�	�	�	�		g	N	;	+	�������cTC0�����tbK8������xW8������|jW>1	������lYG6 ������sZN1"�����cL6������qa       �@ nonfiNORP �? ?us national dairy councilORG�> ton alPERSON�= exer ciseORG�< dailyDATE�; !weektoweekGPE�: +the full  weeksDATE�9 /a subsequent weekDATE�8 ;between an initial weekDATE�7 #three weeksDATE�6 !dos santosPERSON,�5 Wsig nificant betweengroup differencesORG�4 linearORG �3 9highload lowrepetitionPERSON!�2 ;lowload highrep etitionPERSON�1 winwoodORG�0 #adap tationORG�/ bufordPERSON�. csapoORG�- -charac teristicsPERSON	�, esGPE�+ %betweenstudyPERSON�* 9at least several hoursTIME�) #indi vidualPERSON�( dioPERSON�' 'par ticipantsORG�& al  idenPERSON�% #individ ualORG�$ istics egORG�# endurNORP�" %wilson et alORG�! hicksonORG�  decadesDATE� dio’PERSON&� Krt ie aerobicendurance trainingORG
� blyORG� howevPERSON� !rt methodsORG� #femoris  vsPERSON� fourCARDINAL� %selectorizedORG� al  comPERSON� !empiri calORG
� isoORG� %arthur jonesPERSON� fa vorPERSON� crosssecORG� /agonistantagonistNORP� -the final  weeksDATE� preex
GPE� 1the preex haustionORG� xweekORG
� mriORG� 7the successive  weeksDATE�
 1the initial  weeksDATE�	 5an additional  weeksDATE� ;multijoint exercises ieORG
� secORG� )sta tisticallyPERSON� #preexercisePRODUCT� %appre ciableORG� hoursTIME� sys temicPERSON� hormoPERSON�  su persetPERSON� )agonistagonistPERSON�~ em ployedGPE�} !exer cisesPERSON�| detrimenORG�{ !laseviciusNORP�z !dif ficultPRODUCT'�y Mconcur rent training programs ieORG�x benPERSON�w ci  PERSON�v )al ternativelyPERSON&�u Khypertrophyoriented rt programsORG�t poststudyGPE�s involvNORP�r whethORG�q 3crosssectional areaGPE�p brandaoPERSON�o 5muscle hypertrophiesORG�n influNORP�m !a  to weekDATE
�l sysORG�k !rou tinelyPERSON�j 'senna et al  PERSON�i mo dalityPERSON�h ci
PERSON�g ci  toPERSON�f %neu tralizedPERSON�e #exer cisingPERSON�d %conceiva blyORG�c minuteTIME�b /metaanalytic dataORG�a #nonexercisePERSON�` fiveCARDINAL�_ underdeORG�^ %con ceivablyGPE�] %phillips s mORG�\ 'several weeksDATE
�[ limORG�Z -individ ualizingORG�Y !hypertrophPERSON�X +dualenergy xrayPERSON�W per weekDATE�V elliteORG�U rm  PERSON�T %exer cise ieORG�S sys temPERSON�R 7multijoint exercis esPERSON�Q fac torsORG�P simi larPERSON�O #highload rtORG�N 5tra ditional lowloadORG�M -proofofprincipleORG�L )heteroge neityORG�K )discrep anciesORG
�J mvcORG�I max imumPERSON�H ployedGPE�G rt 
PERSON
�F gliORG�E 'more than oneCARDINAL�D ribosoORG�C #rt programsORG�B 1the rd and th weekGPE�A 5the early first weekDATE�@ ;longerterm hypertrophicORG�? +daystoweeks mpsPERSON�> postexGPE�= !longertermNORP�< infuORG�; +shortterm hoursTIME"�: =proteinkg bodyweightmealPERSON�9 exerPERSON�8 mpbPERSON�7 #acute hoursTIME�6 postrtNORP�5 sponseORG�4 1hyperaminoacidemiaPERSON�3 forceproORG�2 -strengthhypertroPERSON�1 imentalORG�0 /subse quent elbowPERSON�/ weekDATE	�. unORG�- 7the mediation ef fectORG�, a weekDATE�+ nuzzoPERSON�* !sociationsORG
�) corORG�( )associ ationalORG�' tenCARDINAL�& %con sideringPERSON�% %the ac crualORG�$ %indi vidualsORG�# bucknerORG�" ar guedORG�! huPERSON�  brilPERSON� =physi ological rationaleORG� robertsPERSON� +rt myofibrillarPERSON� %hypertro phyPERSON� #sar coplasmPERSON� daughNORP� 'rt  jorgensenPERSON� #myofibrilloORG
  H K����������{n]NC	2 �������fwfT@-����������{h^S@)����������|lZG4"���������~p^W��N</=
�
�
�
�
�
�
�
�
�
y
r
h
]
M
C
8
/
#


H	�	��	�	��	��	�	�	�	�	�	�	z�	n	d	Y�	O	E�	7	+	!	������3�yk]N=/%������������~j�aVI=0!����������sj`SH>3%����������rf\PE8X/$�'	�����x�����~twcW�I@6+ �����mcW                                                  izquierdo�linked�	link�
lines�liftervleverage&!intriguing�%interstitial�ischemia�junction�likened�ischemic�kubota�1integrinassociatednintegrityikdaXleydig*	kickisoformsinsulin�leukemia�#interleukin�!likelihood�integral�jnk�lamina�intricate�kelley�linearlyelinearH	linelimits'limitingDlimitedn#limitations
limit
�
limbs�	limb�lim
�likewise�likely#likelihoods!	liketlightload	�#lighterload�lighter	�
light	�lifting	pliftersglifted	6	liftVlifestyle
�	life�	lies�liberally�levels	�
levelAlevyleucine	:
leticlesserh	less.lenient~lengthyelengthten�lengthsD#lengthening>length�
lendslending	lendQlehman�	legs�leg�ledylectionlected�
least`learning �
learn�	leanp
leadsQleading	leadMlbmrlatter@	lats�lations�laterally�lateralis�lateral
later�#latencodingd
lated	�	late�lat�lastly�	lastV!lasevicius+larginine�largest�largerlargely7
large�lar
1)lactaterelated�lactate
lacksClacking�	lack"!laboratory�labeled�lc	kuboj
known �knowledge�	know	<knockout8knocked"
knees�	knee�kinetics�#kinesiology�!kinematics1kinematicskinaseξ|kinaseskinase
kg
_keywords �key �	kept
keoghjuZ	jariX#jyväskyläUjjhK)interpretation=interact4keeping�	keep�keelerikassiano	just�	jury�
jumps	jump�judged		jozo�journal�jorgensen�
jones'jointspecific{joints�
joint�jnajessee�	jeff�
james�j�	ized�
iusca
�ity	Titselfits �
itive	!	itga�!iterations(
itemsI	ited
�itoisting-istics�issuedZ
issue�isotopes	.isotope	0isotonic�'isometrically	isometricisolated	Lisolate�!isokinetic�#isoinertial�isoform�%isoenergetic	jisouisM%irrespective
�irrespec	�!irrelevants	ions�ionMiological�inward�involvingxinvolves �#involvement�involved �involve�involv�#involuntary_%investigator�'investigationc'investigatinge%investigated	#investigate{inverted
�inverse�!invariably�intuitive�intron	�%introduction �intrinsic`%intriguinglyintraweek
%intrasession
'intramuscular�/intrafascicularlyn'intracellular�
intra�	into-'interventions�%intervention_intervals9interval�!interstudy�%interstitium�%interspersed8interset%interpreting#interpreted�interpretOinterplay
�'internationalP)intermittently�%intermittent�#intermedius %intermediate+interindividual
<!interferes%interferenceinterests�'interestingly�#interesting�!interested�interest{interacts%interactions�#interactionp
inter9intensity �#intensities5intensi�intense�intended7intendgintegrins)integrinlinkedintegrin�!integrated	�intake(#insulinlike
   � ����mP0����a> ���~a>����{`B�����jA)
�
�
�
�
�
r
X
=
"
	�	�	�	�	�	q	T	.�����_G, ����kM*�����W:�����qS7����y_F'����|[9�����dA����}bB    �r figNNdobjsee�q outpacingVBGamodincreases�p vastlyRBadvmodoutpacing�o !discrepantNNpobjof�n !tenCDnummoddiscrepant�m #appreciablyRBadvmoddiffer�l differVBPadvclis�k !multijointJJconjisometric�j %requirementsNNSpobjon(�i %%coordinationNNcompoundrequirements �h %sideNNcompoundcoordination�g sideringVBGcsubjdiffer�f #complicatedJJacompis�e linearlyRBadvmodscale�d meaningVBGrelclhypertro�c accompanyVBPccomptend�b icsNNPnsubjaccompany�a !mechanNNPconjphysiology�` !intrinsicJJconjphysiology�_ affectingVBGadvclchanges�^ factorsNNSconjproteins"�] )noncontractileJJamodproteins�\ mattersNNSnsubjchanges�[ #complicatesVBZpobjof(�Z 1!multicompositionalJJconjmultiscale�Y !multiscaleJJamodnature�X realityNNnsubjis�W proteinsNNSpobjof�V crualNNpobjvia�U acNNPcompoundcrual$�T '#associationalJJconjtheoretical&�S #'secondarilyRBadvmodassociational�R #theoreticalJJacompis�Q %contributingVBGaclment�P arguNNPcompoundment�O debatedVBNcsubjresult�N hotlyRBadvmoddebated�M causeNNattris�L toryNNcompoundcause�K contribuNNPcompoundcause�J everRBadvmodeg�I loennekeNNPcompoundal!�H !contentionNNnsubjpassagreed�G #experimentsNNSpobjof'�F ##discussionsNNScompoundexperiments�E riesNNSdobjcatalyzed�D seFWcompoundries�C catalyzedVBDROOTcatalyzed�B #uncontestedJJacompgone�A goneVBNrelclway�@ latterJJnsubjgone �? %particularlyRBadvmodlatter�> argumentsNNSnsubjgone�= norCCccnecessary�< neitherCCpreconjnecessary�; vidualsNNSnsubjgain�: indiNNPcompoundviduals�9 gainingVBGpcompwithout�8 —:punctgain�7 gainNNpobjfor�6 necessaryJJacompis�5 pointsNNSpobjto�4 reducedVBNrelclphenomena �3 argumentNNnsubjpassreduced �2 #independentJJamodphenomena�1 edNNdobjproceed�0 proceedVBPconjlacking/�/ 3%strengthhypertrophyNNcompoundrelationship#�. #%presumptiveJJamodrelationship�- bucknerNNPcompoundal�, holdVBccompgued�+ guedVBNadvclduce�* arJJnsubjgued�) duceVBconjis�( unitNNattris�' +forcegeneratingJJamodunit�& #fundamentalJJamodunit�% sarcomereNNnsubjis�$ additiveJJacompare�# forcNNPcompoundes�" tenetNNpobjon�! basicJJamodtenet�  icatedVBNccompbased� predDTadvmodicated� notionNNnsubjis� equatesVBZaclpremise� largerJJRamodmuscle� premiseNNpobjon&� 3hypertrophyorientedVBNcompoundrt� showsVBZconjhelp� atureNNnsubjshows� literNNcompoundature� mindNNpobjin� trainNNnmodmethods� decipherVBxcomphelp� helpVBROOThelp"� !physiologyNNcompoundresearch� ongoingJJamodresearch� excitingJJamodarea� sponsesNNSnsubjhelp� signNNcompoundvariables� widelyRBadvmodvary(� '+compositionalJJconjultrastructural� +ultrastructuralJJamodes�
 %natureNNdobjinvestigated �	 %investigatedVBNrelclnumber� manNNcompoundstudies� huNNPcompoundstudies � #brilNNPcompoundhypertrophy� primeJJadvclpresented"� 'energeticallyRBconjspatially� spatiallyRBadvmodprime� earlierRBRadvmodoccur � #pertrophyNNnmodhypertrophy�  hyNNPcompoundpertrophy� !adaptationNNnsubjoccur"�~ ologicalNNPcompoundrationale�} physiNNPcompoundological�| robertsNNPcompoundal�{ viewNNpobjin�z reNNcompoundview�y rationaleNNdobjconsider�x considerVBxcompseems�w prudentJJoprdseems�v !oppositionNNpobjin�u phenomenaNNSdobjviewing�t precludesVBZadvclprovide�s !phenomenonNNnsubjbe   � ����fG*
����|[B"����nJ)����oX;#����oN/	
�
�
�
�
p
W
7
	�	�	�	�	�	i	M	3	����wU-����mP1����cD-����zW>�����uU9"	����}`C&�����gI�����qQ(          �r eyesNNSpobjin&�q )%methodologicalJJamodshortcomings�p hinderedVBNROOThindered�o entNNnsubjpasshindered�n presNNSpobjat�m !impossibleJJacompbe�l !experimentNNnsubjbe�k juryNNnsubjis"�j %!conservativeJJamodconclusion-�i -%strengthhypertroNNPcompoundrelationship�h imentalJJamodevidence�g experNNadvmodimental�f !tightRBadvmodcontrolled�e rationsNNSpobjwith�d duFWnmodrations�c fruitfulJJacompbe�b !hardlyRBadvmodmeasurable�a mmNNPpobjof�` standardNNPamoderror�_ quentJJcompoundelbow�^ subseNNcompoundquent�] augmentVBccompassess�\ restVBconjweek�[ !randomizedVBDaclquestion�Z tiveNNPpobjfrom�Y perspecNNPcompoundtive�X questionNNdobjap�W proachVBdobjap�V cleverJJamoddesigns�U !obtainableJJamoddesigns�T unNNPnmoddesigns�S futileJJacompbe�R assignVBadvclis�Q randomlyRBadvmodassign�P %experimenterNNnsubjassign�O 'inconceivableJJacompis�N dependentJJamodvariable�M showVBccompargue �L #alityNNcompoundexperiments�K #problematicJJconjtheory!�J #arguablyRBadvmodproblematic�I !reasonableJJacompis�H %contributoryJJamodcause�G lishVBadvclneed�F estabVBaclevidence�E needVBPadvclis�D correlaNNcompoundtions&�C -experimentalistsNNSnsubjloenneke�B !suboptimalJJamodmeasures�A lectedVBNamodmeasures�@ colNNcompoundmeasures�? nearlyRBadvmodall�> fectNNpobjof�= efNNcompoundfect�< unbiasedJJamodestimate�; collectVBadvclwishes�: typicalJJamodstudies�9 #confoundingVBGdobjassume�8 #residualJJamodconfounding�7 assumeVBPxcompwishes%�6 )datageneratingNNcompoundprocess�5 ideallyRBadvmoduse%�4 %'tweensubjectJJamodheterogeneity�3 !adequatelyRBadvmodaccount�2 wishesVBZadvclis�1 tionshipNNPpobjthan�0 relaNNPcompoundtionship�/ stressedVBNROOTstressed&�. +!implementationsNNSconjapproaches�- !approachesNNSdobjgain�, diatorNNattrbeing�+ mePRPcompounddiator�* againstINprepevidence�) indirectJJamodeffect�( justRBadvmodthat�' jesseeNNPauxdid!�& mediationNNcompoundanalysis�% #causalJJamodconclusions�$ #confoundersNNSpobjfor�# properlyRBadvmodaccount�" mediatorNNpobjas�! modelingVBGxcompsuggested�  nuzzoNNPauxsuggested� lationsNNSpobjof� correNNcompoundlations� %shortcomingsNNSdobjremedy#� #%inferentialJJamodshortcomings� remedyVBadvclsuggested� tributoryNNcompoundcause � %experimentalJJamodevidence� !sociationsNNSpobjas� courseNNpobjof� #noiseNNnmodvariability� relationNNdobjrepresent� corNNPcompoundrelation� insistVBPconjis� !covarianceNNconjbias� unlessINmarkis� attenuateVBxcomptending� tendingVBGaclerror� errorNNpobjdespite� strucVBNcompoundture'� 1variancecovarianceNNcompoundstruc� dominateVBconjaffect�
 #variabilityNNnsubjaffect!�	 !#biologicalJJconjmeasurement� %correlationsNNSnsubjare� lateRBadvmodare� calcuNNpobjto#� #!importantlyRBadvmoddependence � !!dependenceNNdobjsuggesting� !suggestingVBGaclgain� 'weakJJamodrelationships� ationalNNPattris�  associNNPcompoundational� cloudVBxcomptend�~ tendVBccompexist�} !perimentalJJamodfactors�| +straightforwardJJacompis�{ grosserJJRacompare�z %measurementsNNSnsubjare�y typicallyRBadvmodare�x urementsNNSnsubjassess�w #measurementNNnsubjaffect�v icNNPcompoundstrength�u dynamNNPcompoundoutcomes�t closelyRBadvmodfollow�s followVBccompargued    ����cI-����zU9���zeI,����|`H,����mP6
�
�
�
�
�
d
O
1
	�	�	�	�	k	L	#	����gK.�����`D&����hJ����iJ1����w]9�����hM3�����nS>$����t[A                         �q #comparativeJJamodoutcomes�p liftingVBGpcompto�o responsNNSattris�n trialNNpobjfrom�m !subsequentJJamodtrial�l drinkNNpobjthan�k soyNNcompounddrink"�j %isoenergeticNNPcompounddrink�i )proteinmatchedJJamoddrink�h anabolicJJacompbe�g milkNNpobjof�f skimmedVBNamodmilk�e bovineJJamodmilk�d alignVBPconjare!�c turnoverNNcompoundestimates�b plesNNSattrare�a examNNcompoundples�` broadlyRBadvmodare �_ !!phenotypicJJamodadaptation�^ alignedVBNconjare�] !longertermNNpobjin�\ evantJJacompare�[ relNNcompoundevant�Z ableJJamodbalance�Y availVBconjmeasures#�X )siondeterminedNNPamodmeasures�W infuNNPcompoundmeasures�V shorttermNNPcompoundhours!�U 'animalderivedJJamodproteins�T ityNNnmodproteins�S qualJJamodproteins�R )bodyweightmealNNPattrare(�Q )proteinkgNNPcompoundbodyweightmeal�P dosesNNSnsubjare�O permealJJamoddoses�N !estimationNNdobjmaking�M makingVBGpcompin�L isolatedJJamodproteins�K %translatableJJacompare�J #risesVBZadvclstimulation�I #mealinducedNNPnsubjrises�H #mixedJJamodmealinduced�G doseNNpobjbetween�F ingestedVBNamoddose(�E %%doseresponseNNcompoundrelationship�D #stimulatingVBGpcompto �C -hyperinsulinemiaNNpobjwith)�B -resultantNNcompoundhyperinsulinemia�A riseNNdobjtriggers�@ triggersVBZrelclacid�? robustJJconjfull�> 'aminoacidemiaNNpobjto�= 'sensitizationNNpobjin�< knowVBPparataxismps!�; 'carbohydratesNNSconjleucine�: leucineNNnsubjinfluence�9 acidsNNScompoundleucine�8 amiNNROOTami�7 essentialJJpunctpanded�6 liftedVBDpcompof�5 ciseNNPcompoundloads�4 exerNNPcompoundloads�3 pandedVBNadvclduced�2 #methodologyNNdobjusing#�1 #infusionNNcompoundmethodology�0 isotopeNNcompoundinfusion�/ extensiveJJamodreview�. isotopesNNSpobjof�- stableJJamodisotopes&�, /infusioningestionNNdobjutilizing�+ !elucidatedVBNconjduced!�* #undertakeVBxcompchallenging�) #challengingJJacompare)�( -#methodologicallyRBadvmodchallenging�' !admittedlyRBadvmodare�& hoursNNScompoundchanges�% ducedVBDROOTduced�$ !exerciseinNNPconjmeal�# mealNNpobjof!�" !#persistentJJamodstimulation�! itiveNNROOTitive�  posJJnsubjitive� !becomesVBZadvclstimulated!� !!stimulatedVBNROOTstimulated� postrtNNPcompoundperiod� sponseNNattris� thoseDTdobjexceed� exceedVBPccompoccurs� ratesNNSnsubjexceed � /ingestioninfusionNNpobjdue%� 1hyperaminoacidemiaNNnsubjoccurs� stimuliNNSpobjof"� ##synergisticJJamodstimulation� ingestionNNpobjto� acidNNnmodingestion� aminoNNnmodacid� !responsiveJJacompis� %surprisinglyRBadvmodis� !foldVBxcompfluctuates� !fluctuatesVBZrelclmethods� personsNNSpobjin� healthyJJamodpersons� youngJJamodpersons�
 locusNNdobjprocesses�	 accreNNcompoundtion� mpbNNconjmps� breakdownNNcompoundmpb!� !developedVBNccompstimulated#� ##reconcilingVBGROOTreconciling� ultimateJJamodquestion� answeringVBGpcompfor� formativeNNattrbe� uninJJamodformative"�  !%odologicalJJamodshortcomings� methNNconjmuscle�~ nuancesNNSdobjswelling�} #swellingVBGaclcombination�| foundingNNcompoundfactors�{ opinionNNattris�z decreasesVBZccompimply�y implyVBconjhindered�x claimNNnsubjimply�w elementsNNSnsubjpassadded�v ducingVBGamodelements�u forceproNNPamodducing�t credulityNNdobjstrains�s strainsVBZccomphindered
   �� ��������|m]N�>2"�����������/�|lcTD4&K����R���k�����yk_L<. ���~��������r^O�B7*
����\������ncUI<�3'
�
�
�
�
�
��
�
�
�
�
p
`�
W
J�
A
6
)

	�	�	�>	�	�	�	�	�	�	|	j	V��	B	6	'��			F����|q_M=3&	�����������xiYO<0 ���������q^M=+�/�������xfSE8)����������yi��shYJ:/#           %potentiating�%postadaptive~priori
prior
!principlesaprinciple3principal
primeprimary �primarily<pri!precedence[posturalTportions:	plusphosphate�
prado�pkc�pressureg!postulated]pituitaryS#polypeptideKprematurepossess�#phosphatase�playing�!previouslyTprevious	prevents�prevent �!prevailing!pretrainedWpretopost�pretest�#presumptive.!presumablyupresses�
press�%preservationNpresented%present�presence�'prescriptions.%prescription �#prescribing �!prescribed�	pres�+preregistration�/prepostexhaustion�prepostpreparedHpremise#preliminary
 preferredG)preferentially�%preferential
+!preferenceR#preexisting'preexhaustion�preexhaus�#preexercise�
preex,'predominantly#predisposedI!predispose3!predictors�predictor�!predictive.#predictions�!prediction�!predicatedpredic�'predetermined�	pred!precursors�precludes�preclude�precision�preceded�#precautions7pre�pragmatic&	prag^'practitioners%practitioner�practicesVpractice �practical�practic�	ppxy�!powerpoint�%powerlifting�%powerlifters�
power �powG!potentiate
#potentially �potential�	pool�#postmitotic�#perpetuated�'perturbations�#populationsrphysiqueso
placeS!physically3/pharmacologically#potent3phosphofructokinase�'phosphorylase�placebo�ph�+phosphocreatine�'postoperative�plasmaYpervasiveM
piezo(%phosphatases-phosphoproteomic#postworkout
�posture�poststudy�postrt	)postresistance�)postexhaustionN%postexercise	}postex	yposterior�'posteccentric@	post�possibly.possible�#possibilityopositiveapositions�#positioning\position �posited
"posedly�
posed
�pos	 portion�por�!population�!popularity�poorly8	poorapooling	�pooled,ponents�'polymorphisms	�polygenic	�points5pointed�
pointpod�pocket�po^pn�plyingply�ploying
cployed	�plotting6
plots*#plification�	plex�+plethysmography~plethora�	ples	bpldu
plcγM
plays	�players�	play8plausiblyplausible%plausibility�plateaus
�plateau
�
plate#plasmic�plantaris�)plantarflexionplantar�7planningperiodization�planned�planesa	plan�
plain�placingsplacing�placementfplaced�pitching(pipJ
pinto�pictures�physiqueX!physiology'physiological{physioCphysical�
physi�	phys�phy�+phosphorylation<)phosphorylated?'phosphorylate�)phospholipasesD'phospholipaseK5phosphatidylinositolH%phosphatidicBphillips�!phenotypic	_!phenomenon�phenomena�phd�!phasicallyphases	phaseWpertrophypertains �!pertaining�pertain�#perspective�perspec�persons	personnel�personal�persists
�!persistent	"perset�!persession
�permuscle
�!permitting�permitted�permeal	O!peripheral(periods#periodizing
�!periodized�periodize:'periodization�period�
perio�   � ����y_D'
����uV@$����w[="����y`C!����pO1
�
�
�
�
u
^
E
/
	�	�	�	�	|	^	H	/	����aA%
�����aD'�����eF-����mO/�����z_A$����lM+
����}^:#
�����]7               �t highloadNNPconjrm#�s +paringVBGadvclloadindependent%�r +loadindependentJJccompconcluded�q accordNNpobjin�p irrespecJJamodtive�o trueJJamodtive�n heldVBDccompshowed�m #subanalysisNNnsubjshowed�l −INpunctshowed�k zonesNNSpobjin!�j 'correspondingJJamodinterval�i tivelyRBadvmodincluded�h respecNNPconjprovides�g spectrumNNpobjacross�f highJJamodintensity�e somewhatRBadvmodlimited�d fortNNPpobjof�c stoppedVBDadvclperformed�b lowloadNNcompoundtraining�a matchedVBDconjindicate�` advantageNNdobjreporting�_ icallyRBadvmodstudies�^ specifVBPaclprotocols�] levelsNNSpobjby!�\ anciesNNSnsubjpassexplained�[ discrepNNadvclindicate�Z schemesNNSpobjversus�Y lighterJJRamodschemes�X givNNcompoundrange�W !challengesVBZadvclrefers�V questNNpobjin�U wherebyWRBadvmodachieved�T zoneNNPpobjof�S mvcNNPconjrm�R imumNNPnmodmvc�Q maxNNPcompoundimum�P expressedVBNccomprefers�O ployedVBNaclmagnitude�N emNNnpadvmodployed&�M -!evidenceinformedJJamodguidelines�L playsVBZccompbelieved�K believedVBNROOTbelieved�J !refutationNNconjsupport�I !denceNNdobjstrengthen�H confiJJamoddence�G !strengthenVBccompinfer�F 'triangulationNNdobjoccur�E sightsNNSpobjin�D inferVBadvclprovide �C %observationsNNSdobjprovide�B signalNNpobjinto�A #demonstrateVBconjis�@ geneNNpobjof�? fingerNNcompoundgene�> zincNNcompoundfinger�= familyNNcompoundfinger�< gliNNPcompoundfamily�; variantNNpobjin�: intronJJamodvariant�9 %unidentifiedJJamodsnp�8 thelessVBPamodnone�7 sociatedVBNacompbe�6 snpNNPnsubjobserved�5 'polymorphismsNNSpobjfor&�4 'cleotideNNPcompoundpolymorphisms!�3 'singlenuJJamodpolymorphisms�2 underpinVBPccompbecoming"�1 )transcriptomicJJamodprograms�0 clearerJJRacompbecoming�/ becomingVBGccompis�. naïveJJacompare�- whoWPnsubjare�, recruitsVBZrelclstudy�+ everyDTnummodstudy�* aboutRBadvmodevery�) hallmarkNNattris$�( '!nonrespondersNNSconjresponders�' !respondersNNSpobjof�& !socalledJJamodresponders�% enceNNnsubjis�$ quiredVBNacompre�# systemNNpobjof�" #functioningNNconjsystems%�! %%orchestratedVBNamodcoordination�  captureVBconjis� acutelyRBadvmodmeasured� malNNPcompoundcontent� ribosoNNPcompoundmal� innateJJamodresponses� !residentNNdobjhighlights� !highlightsVBZccompis� alsNNPpobjbetween!� 'substantiallyRBadvmodvaries� variesVBZaclfact� !factNNnsubjhighlights� !regulationNNdobjexpect� !polygenicJJamodregulation� expectVBrelclprocess� henceRBadvmodone� systemsNNSpobjin� tegratesNNPnmodsystems� complexJJamodprocess� latedVBNacompre� !integratedVBNamodmps� !headtoheadJJoprdtested� tributingNNattrwas�
 refinedJJamodresponse�	 repairedVBNccompwas� damageNNnsubjpassrepaired� !speculatedVBNccompwere� thXXconjrd� rdNNPnmodweek� !correlatedVBNccompshowed� #daystoweeksNNPcompoundmps� !mediumtermNNnmodmps� waterNNpobjof�  !deuteratedVBNamodwater� fedstateJJpobjin#�~ #fastedstateNNcompoundresponse �} %postexerciseNNPcompoundmps�| !surprisingJJacompbe�{ extentNNpobjto�z erciseNNPcompoundmps�y postexNNPcompoundmps�x hNNPcompoundpostex�w instanceNNpobjfor�v scenariosNNSattrare�u #nonethelessRBadvmodare�t erUHintjloads�s lowJJamodloads�r namelyRBadvmodoutcomes
� �� ������l[J.6������oaG;��@������MB�5�,���������|paUG=Y/������vPC:	
�
�
�
�
�
�
�
��
u
R
B
9
-

	�	�	��	�	�	h	^	S	G	8	(		������scR=#������m�dSE;)��������}�k]C:&���������t��r`O9.$���������fULC�����������tcL<
���� 1autocrineparacrine�	behm�halflives�  �lynn�;lowload highrep etition2los ing �-longmusclelength;longerterm hypertrophic �!longerterm �1longermusclelength%longermuscle%longerlengthlohmannvlinear4lim �leydig�
lehman{latY!lasevicius �l  °  °DkuboBkelley�
keelerkassiano0ejozo grgic cody t haun eric r helms stuart mwjnk�jjh�%jeff nippard5jari ylänne ju chen�%james wrightt)james steele  y%james steele#james pvistics eg$isoinvolv �!interstudy<Ainstitutional review board#inoki et al}	infu �
influ �-individ ualizing �#individ ual%%indi viduals �#indi vidual)imental �ilr
igfe'derecruitment�anderson�dozens�-fasttwitch fiber�fourth�/bodybuildingstyle�!fasttwitch�1calcineurin camkii�%after  weeks�3aquaporin aquaporin�7cellular hydration ie�)hubal  proposelhu �+httpsosfiowdjxgq'httpsosfiouvdm-httpsosfioahjnf
c
howev
hours
hormo#highload rt �9highload lowrepetition3hickson!)heteroge neity �hek|7height metaregression`hamilton �'halfrom forceT)half a millionD	halfd	grgicp&Qgradepro httpswwwgradeproorg  this3goto@gli �+george brooks  z)frey et al  tok	fourforcepro �	five �%fitomics llc �	firstsfilamince#femoris  vsfedP;faculty of sport health$/faculty of health �fac tors �fac �fa vorAexercisespecific techniqueC	exercisespecific kinematics!Gexercisespecific instructions7exercisespecific body-exercisespecific#exer cising �!exer cises �%exer cise ie �exer cise=	exer �-evanston il usa
 �es,!eric helmsuE�environmental science auckland university of technology private �english-
endur#!empiri calem ployed �ellite �
eimd
n	eimdq9dystrophinglycoproteinN0edynamometerselectric goniometerstensiometersV+dualenergy xray �!dos santos6dividual]?diseasespecific mortalityF)discrep ancies �)dis tal muscleYdio’dio(!dif ficult �"Gdiacylglycerol kinaseξ dgkξ]detrimen �!Gdepartment of health scienceszdecades +daystoweeks mps �
daugh �
daily<csv9
csapo.3crosssectional area �crosssec"Gcostamerebased mechanosensors`cor �cooccurm%Mconcur rent training programs ie �%conceiva bly �%con sidering �%con ceivably �/coactivating teadT!Gclusively resistance machinesZ+cjun nhterminal�ci  to �	ci   �ci
 �-charac teristics-chang es �	casahcanadianAcanada �calciumca�buford/buckner �'Sbronx ny usa school of sport health|	bril �brandao �brad"bly!birmingham �)betweensubjectW%betweenstudy+;between an initial week8ben �bayesian'
august14kauckland  new zealand department of kinesiology �%at least two.9at least several hours*%at least one/)associ ational �%arthur jonesar gued �%appre ciable
ankrdW	ankit8 Eandroulakiskorakakis james p!an hourk5an additional  weeks	allcauseE/alfred goldberg  J2galabama sports performance research institute �)al ternatively �!al  keeleral  iden&al  first[al  comald%aktmammalian�/agonistantagonist)agonistagonist �#adap tation0#acute hours �'Sabstrackr httpabstrackrcebmbrownedu2a week �/a subsequent week9/a few weeks later7
a day�1a couple of months�!a  to week �
� ���aQ�]��>����������}nbnb����YD8`C�%��P��~��������h^O4�����lXJ7&�������������$��~���|�F�]��0k�i�u�eT������E8�W/#
P������������8zfY;�/#Q%)}
"���5s��������F�yq��i�V> EG	�	�	�b	�	ob	a	M	C	8	.�		�`���������]      myotrauma�%martinezcavaA	mapk�mgf�max imum �)matt alexander�mass lbmp mapk�%metaanalyses='membranebound{melbourne)mechanosensorsK)mechanosensingC'mcmahon et alf)rapamycin mtor�mrf�/myf myod myogenin�!myonuclear�myonuclei�	mrna�!myonucleus�#mitotically�)noncontractile�morgan�Amuscle crosssectional area�october�igfHs�5musclespecific force)musclespecific~mmolkgy1mmolkg preexercisex
igfea�7pcr resynthesizes adpuU whypertroXrossmonthsp%nearlifelongo
igfeb�	ppxyinonmuscleg'igfeb  levels�!hypertroph �%hypertro phy �psk_	mtor^$Khypertrophyoriented rt programs �
mtorcZ
igfec�X	ptenVmtorc yapU#indi vidual)imental �ilr-igf testosterone�;ial crosssectional area �#protein·kgG#mtorc hippoB
nonfi@involv �!interstudy<Ainstitutional review board#inoki et al}	infu �
influ �-individ ualizing �#individ ual%%indi viduals �'par ticipants'jjh�iso$Krt ie aerobicendurance training!rt methods� jnk�james pvistics eg$preex
%james wrightts%james steele#mri3 th5jari ylänne ju chen�)james steele  y;multijoint exercises ie
keeler)%jeff nippard#preexercise� skelley�kassianopoststudy �kuboB �5muscle hypertrophies �latY!rou tinely �x sl  °  °Dmo dality �%neu tralized �minute �/metaanalytic data �#nonexercise �
lehman{ �%phillips s m �!lasevicius �s �per week �	rm   �leydig��7multijoint exercis es �� silohmannvlinear4lim �-proofofprinciple �mvc �ployed �	rt 
 �  �yearswolf  Vwilk%whaley et alK1wgt cmjsprint timeL-werkhausen et alN	weeksQ
weekly4	week �)vic  australia �valamatos?;upperbody muscle groupsnupperbody[un �u  °  °Etwo1torgluteus maximusItidybayes:#ticipant  lMticipant
C	threea  :thigh crosssectional area1the rd and th week �-the past century �7the mediation ef fect �*Ythe international prospective register,5the early first week �%the ac crual �#terventions\ten �!tal muscle �  Ssubse quent elbow �-strengthhypertro �/stasinaki et al  j%sportsdiscus&?sport victoria university~sponse �!sociations �J�social sciences solentnardone�igf mrna�9less moderate  seconds�5longer than one hour�#rm  minutes�#multipleset�#hypertophic�#microtrauma�
prado�maimum rm�kubota�!maeo et alh	lynn�;lowload highrep etition2los ing �-longmusclelength;longerterm hypertrophic �!longerterm �1longermusclelength%longermuscle%longerlength0ejozo grgic cody t haun eric r helms stuart mw3rt exercisespecific'rt  jorgensen �rrsrromRroberts �rm
GrmHriboso �!retic ulum �%recent years('pubmedmedline0=proteinkg bodyweightmeal �3prom groupcondition53preex isting muscle �postrt �postex �=physi ological rationale �phillipsx+pereira et al  	pereira
pedrosaOpallarésopallaresJ+pak mw mc rb appak!only threeonly oneonesize)one>#onciliation �
nuzzo �nonwarmup_-nonstrict strictnogueira!neutralarmi#myofibrillo �!myofibril
 �mvicPmpvFmps �mpb �'more than one �3milo wolf patroklos %micro scopic �)metaregression^3metaanalyses prisma+
� �|�����	�������������y�k\K@8-%���������:+��q`QE�6*
������������yl_UJ=0&(������pbSC70"�����|����~pg
NWL@�7.%
��������[��ta�SB5'���������~u�5jbVcWJ@6�*������h�������v�h[J8'��
��
�
�
�
�
�
�H
�
�
�
x
m
c
V
I�
>
3
&


	�	�	�w	�	�	��	�	�	�		s	f	U	F	6�	)		���������w8h\J6�*�ososph  )phospholipasesD%phosphatidicB'posteccentric@)phosphorylated?+phosphorylation<pn�organisms�!noteworthy8ip
notedH	noteunotably�notableB!perception�
opted�al� �phosphorylate��postworkout
�posture�poststudy�postrt	)postexhaustionN%postexercise	}postex	yposterior�	post�possibly.possible�#possibilityopositiveperforms�obliquesm%partitioningF!pectoralis-#overtrainedosmolytes�optimizes�opening�3oxidativeglycolyticvosmotictovaries.paracrinepeptide
overt�?nuclearcontenttofibermass�partly�octoberTotherwiseEobscurin	partially	ohno�oxidative�pcr�-occlusionrelated�occlusion�occluded�patients�oxygen�%Moxidemetalloproteinasehepatocyte�numbers|!parametersPnuclei&nucleus	pair�!opposition�opposite�opinion�'operationally�	operbopenly�	openK	oped
�onymous�onwards
onset�	only�ongoingonesize	ones�one=#onciliation?	onceuon&	omniological�
older�ogeneity�	oftenoffers�offered�off�ofodse!odological	 odized1odization�occurs�occurring�#occurrences�occurred�
occur�occupy�occupied�!occasionalaoc�obtained�!obtainable�obtain�observed�%observations	�!obligatory�objective�oa�
n’t�ny�
nuzzo�#nutritional�nutrition�numerous�number�num
]nuclearonuances�nuancedz	nsca �novicefnoveltyj
novel�+notwithstandingPnotionnoting �perienced|	peri�perhaps?!performing/performed �#performanceNperformYperforperfectly�pereira�!perceptualT#perceptionsU!percentageperceived
Qper �people�pendingkpendent%peerreviewedZ	peerpedrosa�
pedro�
pedalpearson�peakingv	peak
pause�paucityKpatterns^patternOpatroklos�pathways
pathway^	path�patentXpatchwork�	pastDpassiveX%particularly?!particular
�#participate�%participants�#participant�particikpartials$partial�	part
paring	�
pared
nparameter@parallel�parmpapers�
paper	pantlpanded	3pallarésspallares5pakbpairing5paired+	pain�	pact�package�pacity:p�
oxide
�ownloverviewt%overtraining
�overtrainoversim�%overshadowed=overreachingovertrainingX%overreachingJoverreach
�overly�!overloaded�overload�overlap�#overheadarmRoverheadU'overestimatedrovercome\overall�	over�outward�outside�outputs�output%outperformed�outpacingqoutliningIoutlinedoutcomesOoutcome �out�ous�our�othersK
other�!ostensibly�osfNoriginalWorigin�!organizing�organized+%organization=organiza!organelles�organelle�	orga�
order,'orchestration �%orchestrated	�oranchukEorUoptionu!optimizing\optimized>optimize�!optimisingDoptimal� opted�pocket�opassively�� portion�)overexpressing�� 	plduparalogueQ   phospholipaseK   ~$ ����rQ)����yY=����}]4�����iP0����ya:
�
�
�
�
r
U
4
	�	�	�	�	s	Y	=	#	����sU:  ����lH)����~_<�����[:����y\@!�����v`G,����hQ4�����iD$                                �r magneticJJamodresonance"�q %strengthenedVBDccomprevealed!�p !!stratifiedVBNconjidentifies�o gradedVBNxcompfound�n paredVBDrelclstudies�m cerNNPpobjto�l tweenNNcompoundvolume�k driverNNpobjas!�j !!identifiesVBZROOTidentifies�i elliteJJamodresponse�h satVBDconjshow�g signalingVBGcompoundmps�f #tracellularJJpobjin(�e +volumedependentNNcompoundincreases�d volumesNNSdobjploying�c ployingVBGacladvantage�b quantifyVBrelclstandard�a expressVBrelclways�` viableJJamodways�_ kgNNSdobjload�^ berNNPpobjincluding�] numNNPcompoundber�\ waysNNSpobjin�[ speakingVBGcsubjrefers�Z blocksNNSpobjwith#�Y !%strategiesNNSdobjimplementing�X %implementingVBGpcompby�W sessionNNpobjwithin�V repetiJJcompoundtion"�U %accomplishedVBNccompsuggests�T vantagesNNSdobjproduce�S adNNcompoundvantages�R atetohighNNcompoundloads�Q perceivedVBNamodeffort�P #ratingNNconjdispleasure�O #displeasureNNdobjproduce%�N !#discomfortNNcompounddispleasure�M !tendsVBZccompconsidered�L lowloadsNNPpobjwith#�K #!furthermoreRBadvmodconsidered�J temNNPconjjoints�I sysNNPcompoundtem!�H 'neuromuscularNNPcompoundtem �G 'taxingVBGconjtimeefficient�F 'timeefficientJJacompis�E %prioritizingVBGpcompto�D !comparableJJamodmuscle�C achieveVBROOTachieve�B sideredNNPattrbe�A exercisNNPcompoundes�@ singleJJoprdnoted�? #freeweightsNNSnmodarea!�> geneticsNNScompoundmodality�= !submaximalJJamodrm&�< +#interindividualJJamodvariability�; %respectivelyRBadvmodcurl�: armNNconjpress�9 ≥XXcompoundrm�8 indicatedVBDconjappears�7 thresholdNNattrbe�6 beforeINprepshowed�5 !terminatedVBNamodsets�4 tensityNNpobjin�3 !attributedVBNccompshow�2 +inconsistenciesNNSpobjto�1 #larNNPapposdifferences�0 simiNNPcompoundlar�/ ditionalJJamodlowload�. traNNPnmodlowload�- #traditionalJJamodrt�, milderJJRamodform#�+ %#preferentialJJamodhypertrophy�* inducesVBZccompposited�) bfrNNPcompoundtraining#�( #restrictionNNcompoundtraining�' reportNNcompounded�& iiNNPcompoundfibers�% targetingVBGaclhighloads�$ highloadsNNSpobjwith�# %typespecificJJaclfiber�" positedVBNROOTposited �! !discussedVBNconjconsidered!�  #!preliminaryJJoprdconsidered� !priorRBadvmodpotentiate!� !!potentiateVBadvclinitiating$� -strengthorientedJJamodtraining� blockNNpobjwith#� /hypertrophyorientNNcompounded� !initiatingVBGpcompof� amplifyVBconjranges!� %intrasessionNNcompoundbasis� intraweekNNpobjon� anabolismNNpobjon� %amalgamationNNnsubjhave� ablyRBadvmodhave� conceivNNpobjon� 'contradictoryJJacompis� pathwaysNNSpobjof� kinaseNNcompoundpathways� !selectiveJJamodactivation� combinaJJamodtions� failsVBZconjstudied&� -!proofofprincipleJJamodstandpoint� insightsNNSdobjprovides�
 erateNNadvclstudied�	 modNNcompounderate� heavyJJadvclparing� termsNNSpobjin� studiedVBNROOTstudied� gapsNNSROOTgaps� variancesNNSdobjsampling� neityNNdobjrevealed� heterogeJJcompoundneity� !colleaguesNNSconjlopez�  lopezNNPpobjof#� %networkNNcompoundmetaanalysis�~ !correspondNNcompoundence�} ateVBDnmodence�| moderPRPpobjbetween%�{ %moderateloadNNcompoundcondition�z equatingVBGconjcomparing�y poolingVBGcsubjshows�x #×JJamodrepetitions�w lightJJamodload!�v lightloadNNcompoundtraining�u ≤NNPcompoundrm   } ����bB �����dF.�����pW?�����dA ���gL3
�
�
�
�
q
Y
@
	�	�	�	�	z	\	>	"	����dB#����kJ-�����qN2����wY2����kO-����vP6�����iC$����iN5                        �o umeNNconjfrequency�n volNNcompoundume�m interplayNNnsubjis�l manageVBxcomphelp!�k !#standaloneJJamodalterations�j equatedVBNadvcltrained�i daysNNSconjtrained�h %irrespectiveRBadvmodfound�g #frequenciesNNSpobjversus�f %metaanalyticJJamoddata#�e !tweengroupNNcompoundprotocols�d lyXXadvmodsensitive�c callingVBGaclfer#�b #nonexerciseNNPcompoundcontrol�a difNNPcompoundfer�` oxideNNpobjby�_ deuteriumNNcompoundoxide�^ routinesNNSdobjperformed�] fiveCDnummodtimes#�\ 'volumematchedVBNamodfrequency"�[ )moderatevolumeVBadvclsupport#�Z '!volumeequatedJJamodconditions�Y aleNNnmodresearch�X rationNNcompoundale�W soundJJamodale�V seeminglyRBadvmodsound�U #frequentJJamodstimulation�T speculateVBxcompled�S untrainedJJamodviduals�R timeframeNNpobjover�Q persistsVBZrelclresponse �P #displayVBPrelclindividuals(�O /#resistancetrainedJJamodindividuals�N truncatNNPcompounded$�M #postworkoutNNPcompoundduration�L returnsVBZconjremains�K elevatedJJacompremains�J quencyNNPpobjfrom�I freNNPcompoundquency&�H !!tificationNNPnsubjpassconsidered�G !quanNNPcompoundtification�F sessionsNNSpobjof�E readjustVBconjlimit �D #incrementalJJamodincreases�C limitVBxcompbe �B /supercompensationNNpobjfor�A eryJJamodperiod�@ recovJJamodery�? activeJJamodery�> tolerableJJamodvolume�= highestJJSamodvolume�< overreachJJamodphase�; culminateVBROOTculminate!�: #programmingNNnsubjculminate�9 empiricalJJamodevidence�8 athleteNNpossrange�7 constantJJacompremains�6 receiveVBccompconsider�5 opedVBNamodmuscles�4 welldevelJJamodmuscles�3 velopedNNPcompoundmuscles�2 underdeNNPcompoundmuscles�1 cyclesNNSnsubjreceive$�0 )specializationNNcompoundcycles�/ inglyRBadvmodconsider �. )underdevelopedJJamodgroups�- vanceNNpobjof�, releNNcompoundvance�+ !particularJJamodvance�* trophicJJamodresponse�) scriptionNNattrbe�( minimumJJamodscription�' seemVBccompculminate�& 'approximatelyRBadvmodsets'�% #%explorationNNnpadvmodovertraining!�$ %warrantsNNSdobjovertraining�# igatingVBGaclmit�" mitNNPadvclhelp�! ceivablyRBadvmodincreases$�  'progressivelyRBadvmodincreases!� #periodizingNNcompoundvolume� posedVBNrelcltime� guideVBadvclneeded� startingNNcompoundpoint� serveVBROOTserve� itedVBDconjapplied� limNNPnsubjited"� 'incrementallyRBadvmodapplied� ualizingVBGamodvolume� individNNPpobjto"� ##ancetrainedJJamodindividuals� resistVBpobjin'� !%hypertrophNNPcompoundrelationship � )hypotheticallyRBadvmodvary� lifestyleNNconjgenetic� geneticJJamodfactors� #undoubtedlyRBadvmodvary� !determinedVBNconjhave� %overtrainingVBGpcompto� plateauNNadvclconfer"� %increasinglyRBadvmodadditive�
 conferVBccompis�	 hormesisNNpobjof� ceptNNpobjwith� curveNNdobjfollows� ushapedJJamodcurve� invertedVBNamodcurve� followsVBZccompis� ferNNpobjin#� %extrapolatedVBNadvclcomprised� binedVBNdobjcom�  comprisedVBDccompnote� showingVBGaclstudies�~ ingsNNSnsubjare�} greatJJamoddoses�| %unresponsiveJJoprdappear�{ tocolsNNSnsubjenhance,�z %+sorptiometryNNPcompoundplethysmography�y %abNNPcompoundsorptiometry�x xrayNNPnpadvmodisolated�w !dualenergyPRPnmodxray�v sensitiveJJamodmeasures�u !ultrasoundNNdobjimaging�t imagingVBGaclresonance�s resonanceNNapposmeasures
   �	 {obYMC,����������$tdQC6&��������{nbTF8)���������q`RB5&���������qcTG9(�
�
�
�
�
�
�
�
�
�
x
k
`
Q
G
;
.


	�	�	�	�	�	�	�	�	�	�	~	r	g	\	N	<	,			����	��������yj_RI?4(�����������|m_UI</!����A�����paRD4 ���������{l[MA/���������rbTE9,��3�������yk]PB5'��������}n^N�����   rad·s−�regainpproducingLrefutingArebound!regulatory�!regulators�regulatorm!regulation	�!regulatingregulates�regulated�regulate�regularly�!regularityIregular�!regression%registration�!registeredJregisterRregions�regional�regimes�!	regimentedregimensyregimen�%regeneratione#regeneratedi!regenerate'!regardless�regarding�regarded�regard�refutes�!refutation	�reflect|!refinementRrefined	�refers�referring �referred �!references�/referencecitationreferenceyref�redundant!reductions�reduction�reducingSreduces�reduced4reducep
redox�recurring�%recuperative?%recuperationrrectus�recruits	�#recruitment�recruited�recruit�)recreationally�%recreational�recoveryF
recov
�#reconciling	)reconciliation reconcile�#recommended�+recommendationsy)recommendationd!recommenda
�recommend;recognize�recipientmreceptors�receptor�recentlyVrecent�receivedrreceive
�rec>reasoning�!reasonablyF!reasonable�reasonrealizedrealityXreadjust
�readingreadersyreader�	readnreactive�reactions�reactionzreachingBreaches+
reach[re�rd	�rbg
rauch�	rats,ratios!rations�rationale�ration
�
ratiorating
Pratherx
rates		rate�rat�rapidly7
rapid�rapamycin�ranges)ranged-	range7randomly�!randomized�random�ran�raises�
raise�rabbits�r�quoCquired	�quiescent�questions�!questioned�%questionable�question�
quest	�
quent�quency
�quantitymquantile8quantify
b	quan
�quality}	qual	S!quadriceps�q�putting*putative9put�pursuedbpurposes�purposelypurpose]purelyG	pump�pulsatileU
pulls�pulleys`pulldownspulldown�	pullb'pubmedmedlinefpubmed�published7%publications{#publicationptk+	pten^psk prudent�proximity �#proximitiesKproximalproxies�providing�providesmprovided�provide!proven�
prove;protocolscprotocolFproteomic6#proteolysis�#protein·kg�proteinsW)proteinmatched	i'proteinkinase�proteinkg	Qprotein'!protective�prosperoS#prospectiveQ!propulsive�proposes�proposed �proposeKproposal�%proportional�!proportion�!properties4properly�
proper)-proofofprinciple
!pronounced�promoting�promotespromoter�promotedpromote	promise	prom�!prolongingnprolongedW#prolinerich�'proliferations#proliferaterproject�'progressively
�progressAprograms!#programming
�!programmedprogram profiles�'professionals.!production�productPproducesTproduced�produce�processes/processVproceed0problems0#problematic�problem%probablyuprobable>#probability3/probabilistically�	prob�proach�proMprivate�prisma�priors!rads·s−�'psychological�rel	[#reiterating 'reinforcementjreinforce�
reimarecoveredu   ) ����u\?*����yV9����z^5����lR6�����fA 
�
�
�
�
o
Q
9
#
	�	�	�	�	o	O	/	����yZ:�����wU:�����e? ����sN8$����tT:����rV5�����iL2����mM)                                   !�n /intrafascicularlyRBamodbody�m terminateVBPrelclfibers�l inseriesNNScompoundfibers�k neuronsNNSpobjby�j motorNNcompoundneurons�i !innervatedVBNrelclfibers �h %subdivisionsNNSdobjcontain$�g -threedimensionalJJamodresearch�f carryVBaclability�e diverseJJamodability�d tachmentsNNSpobjat"�c #establishedVBNccompindicates�b pullNNpobjof�a !planesNNSdobjmultijoint�` pulleysNNSpobjincluding�_ cableNNcompoundpulleys�^ volvesNNSpobjin�] selectionNNROOTselection�\ cularNNPamodfailure�[ encedVBNaclinflu�Z influJJattrbe�Y %machinebasedJJamodexercis�X !secsNNSdobjperforming!�W minutesNNSnsubjpassemployed�V lastVBROOTlast�U ruleNNcompoundperiods�T producesVBZdepare�S reducingVBGccompshowing!�R #tematicallyRBadvmodreducing�Q lendVBPccompenhance$�P !!efficiencyNNnpadvmodcontrolled!�O !workoutNNcompoundefficiency"�N %!preservationNNdobjfacilitate�M ionNNpobjin�L fashJJamodion"�K %consistentlyRBadvmodtraining�J tinelyRBadvmodemploy�I %rouNNPapposbodybuilders�H erliftersNNSpobjto�G powNNPcompounderlifters�F protocolNNpobjof�E adaptVBadvclability�D 'considerationNNnsubjis#�C 'workoutsNNSnsubjtimeefficient!�B bufferingNNcompoundcapacity�A gleVBROOTgle�@ sinNNpobjin�? flyVBnsubjmean�> chestNNpobjin�= titionNNcompoundreduction�< repeNNPcompoundtition�; dropoffNNnsubjmean�: sennaNNPcompoundal�9 #singlejointJJamodexercise�8 multiNNnmodsenna�7 impairedJJacompis�6 dalityNNconjtype�5 !influencedVBNccompis�4 #conceivableJJacompis�3 adultsNNSpobjin"�2 !moderationNNcompoundanalysis�1 !volumeloadDTdetanalysis�0 equateVBadvclperformed!�/ #tralizedVBDrelcldifferences�. neuNNPnsubjtralized�- uteNNPcompoundperiods�, minNNPcompoundute �+ %pairedVBNccompdemonstrated�* longoNNPcompoundal�) cisingNNPadvclperformed�( !peripheralJJamodfatigue�' blyRBadvmodexplained�& conceivaNNcompoundbly�% #centrationsNNSpobjdespite�$ multisetJJamodexercise�# minuteNNpobjversus�" bluntedVBNccompfound�! rpsNNPconjpsk�  pskNNapposphase� mckendryNNPcompoundal!� %overshadowedVBNadvclappears� tuationsNNSpobjof� flucNNPcompoundtuations� relevanceNNpobjon� doubtNNdobjcasts� castsVBZadvclulated"� )concentrationsNNSdobjresting"� )hormonalJJamodconcentrations� restingVBGpcompin� icalJJamodcrit� critNNattrbe&� +#exerciseinducedJJamoddevelopment� !regulatingVBGpcompin� %fluctuationsNNSnsubjplay � %systemicJJamodfluctuations� ulatedVBNROOTulated� specNNnsubjulated#� #periodsNNScompoundresearchers� !elevationsNNSconjfactor� factorNNdobjshowing�
 #insulinlikeJJamodfactor�	 %testosteroneNNnmodfactor#� %hormoneNNcompoundtestosterone&� !+prevailingVBGamodrecommendations� entedJJamodtraining$� )hypertrophyoriNNPnsubjtraining� secondsNNSpobjto� organizaNNcompoundtions!� !henceforthRBadvmodinfluence� intersetJJamodperiod�  ferentJJamodexercises� !distributeVBadvclcap �~ !persessionNNcompoundvolume�} capVBxcompbe�| !recommendaNNcompoundtion�{ spreadVBxcompbe!�z permuscleNNcompoundtraining�y %infrequentlyRBadvmodtrain�x %distributionNNpobjon�w cyNNPnsubjbe�v frequenNNPcompoundcy�u scrutinyNNpobjin�t wastedVBNamodsets�s exceedsVBZcsubjsuggests�r iuscaNNPpobjof�q plateausNNpcompbeyond�p inationNNnsubjseems   � ���xW<�����tU8�����iG,�����`<�����wR-	
�
�
�
�
|
`
>
%
		�	�	�	~	b	G	.	����qV8����oS6�����x[@(����jT;!�����mF#���o[?  ����nP3�����hO0         �n validatedVBNoprdtermed�m )selfdeterminedJJamodrm�l termedVBNpcompto�k predicJJamodtion�j #correspondsVBZpobjof�i develNNdepoped�h rirNNPcompoundmethod�g anothUHdepcompleted �f completedVBNadvclattempted�e !definitelyRBadvmodachieve�d !nextJJamodrepetition�c !finalJJamodrepetition�b endpointNNpobjto�a !prescribedVBNamodform�` deviationNNpobjwithout�_ porNNcompoundtion�^ !attemptingVBGpcompdespite�] #entireJJamodmusculature�\ cohesiveJJamodstrategy�[ !collectionNNdobjply�Z plyVBattris"�Y )considerationsNNSdobjapplied$�X !)anatomicalJJamodconsiderations�W attentionNNnsubjpassgiven!�V #ternativelyNNPadvmodrotated�U reinforceVBadvclperformed �T regularlyRBadvmodperformed$�S !freeweightNNPcompoundexercises�R velopmentNNPdobjmaximize�Q mingNNpobjof�P earliJJamodstimulus�O inducedVBDconjwas�N learnVBxcompeasier�M curlsNNSpobjin�L midJJdepfound�K poststudyJJpobjat�J legsNNSconjtrunk�I trunkNNpobjin�H #delayedVBNamodhypertrophy�G !chillibeckNNPcompoundal�F novelJJamodstimuli�E recurringVBGamodstimuli�D liberallyRBadvmodrotated�C plexNNcompoundexercises�B skillsNNSpobjof�A helpsVBZccompmakes�@ pressesVBZccompmakes�? rowsNNSdobjsquats�> squatsVBZccompkeep�= involvVBNoprdkeep�< keepVBadvclmakes�; makesVBZROOTmakes�: logicallyRBadvmodmakes�9 fixedVBNamodexercise�8 slightlyRBadvmodmore�7 gainedVBDROOTgained�6 selectedVBDrelclselection"�5 'autoregulatedJJamodselection!�4 %rauchNNnpadvmodinvestigated �3 %undeterminedJJacompremains�2 aticJJamodapproach�1 differedVBNconjachieved�0 whethNNPapposdatabase�/ databaseNNpobjfrom�. #choseVBDrelclapplication�- rotationNNnsubjhad$�, -sessiontosessionJJamodrotation�+ bazvalleNNPcompoundal�* rotatedVBNpcompto�) !frequentlyRBadvmodrotated�( tyNNPdobjinclude�' varieNNPcompoundty�& crudeJJamodestimate'�% !%cumferenceNNPcompoundmeasurements �$ %cirNNPcompoundmeasurements�# #midwayRBadvmodfreeweights�" switchingVBGpcompto�! nhoutsNNScompoundal�  aereNNcompoundal� !schwanbeckNNPnpadvmodplay� ticNNcompoundeffect� synergisNNPamodtic� 'complementaryJJacompbe� !advantagesNNSnsubjseem� agonistNNpobjof� !respondingVBGaclcor!� !stabilizerNNcompoundmuscles"� #!stabilizersNNSconjsynergists"� !#synergistsNNSdobjstimulating� expenseNNpobjat� howevVBZaclability� affordVBconjplay� freedomNNpobjof� !modalitiesNNSpobjof� #selecVBNccompdetermining!� #determiningVBGadvclmaximize!� %lengthtenJJamodrelationship� !seatedVBNdobjperforming� !demonstratJJcompounded� placingVBGpcompon�
 focusVBccompsuggests�	 targetedVBNamodexercise� routineNNpobjin� sionNNcompoundresults� extenJJamodresults� #handNNcompoundperformance� vastiJJamodmuscles$� )preferentiallyRBadvmodinferred� lyingVBGamodtension� brandaoNNPpobjof�  fosterVBacleffect� lectionNNdobjperformed�~ nonvariedJJamodlection�} #extremitiesNNSpobjof�| periencedVBDccompfound�{ costaNNcompoundal�z ilarlyRBadvmodstudy�y #simNNapposperformance�x #uniformJJamodhypertrophy�w lungeNNpobjof�v smithNNPcompoundmachine�u fonsecaNNPcompoundal�t ampleJJamodfonseca�s combiningVBGcsubjenhance!�r #incorporateVBccompspeculate"�q 'architecturalJJamodvariances�p #interactionNNpobjof �o !nonuniformNNcompoundmanner
�"=0"���������r^~eYK</#n�	���t�������z���ugXK<,�����������}ocT=/Q ���������}�mcVE3'

�
�2�
�
�
�
�
�
�
��
�
v
g
Z�
L
<
2�@
&

	�	�	�	�	�	�	�	�	�	��	�	�	~�	v	lk	_	Q	C	:	-�				�������������tj4`TF:1&���O��"���R��mq^�cTF��9l) |�E��������vgZK=5-!��������aSD4* �����������ufWF4 ���������`4%   %satisfactorytsallesrrhomboidsSsartorius;scattered4scapula*!repetitiveretain�reside�#replacement�routinelydresarevisedN� 
roles�robustly	rheb�scavenger�#resynthesis�'resynthesizes�'resynthesized�schwannsaying�reliesq�ros�repairxrespondedtremove�ted)repartitioningLrepeated?	reps�te� rreversal�schematic�I 	regre)semitendinosus4+semimembranosus5)selfdetermined�%selectorized0#selectively"selective
selection]selected�
selec�segmentsK	seep�	seen�
seems]seemingly
�	seem
�seekingn	seek#seeing�see&sectionssectionallsection	secsX!secretionsVsecretionHsecretedHseconds!remodeling�	rely�secondary�#responsible�secondly#secondarilySsecond�!seccentric�sec�seated�seasonsearchingEsearchedhsearcheseDsd�scrutiny
�scrumming#scripts�scription
�screeningoscreenedxscores�scopicXsclerosis6!scientific �sciences�scienceL
scienF!schwanbeck�schumann�school�scholarly!schoenfeld@schemes	�schedule�scenarios	vscenariohscarcity<scarce �
scant*
scale�say�saw/	satoLsatellite6sat
h%sarcoplasmic�!sarcoplasm�!sarcomeres�sarcomere%!sarcolemma�
sarco�sar�santos(sampling�sample�	sameR	saidsafety�	safej!sadacharan�sa�sJrunning�run[	ruleU
rugby"rtinduced�rtd�rtrrsrrrps!	rows�rowroutines
�routine�rouIrotation�rotatingrotated�	romsrrom9	role9rohatgi�rodents�robust	?roberts�rmsxrm�
rized�	risk �
rises	J	rise	Arir�rigorous	riesEribosomes�riboso	�	rheaa!rewrapping-rewarded�reviews�reviewing6reviewedX
review!revealed�reveal�returns
�returnedv!reticulums�reticulum�
retic�resultsresultingresultedFresultant	Bresult%restrictionsd#restriction
(resting	rest�!responsive	responsesresponse�respons	o!responding�!responders	�#respondents�respond�%respectively
;!respectiveqrespect�respec	�resonance
sresisted�/resistancetrained
�!resistanceresist
�residues�residual�resident	�reserveJ#researchers �researchLrequiring �requires �%requirementsj#requirement�required�requirerequest�!representsRrepresentreportsMreportingHreported�report
'#replication�replicate�#repetitions�!repetition4repeti
V	repe<repaired	�rep�	rent
#renaissance�removedGremoval"remedy�remainsremaining�remainedHremain7relying�reliable�relevantrelevancereleasers!releasedrelease�	rele
�!relatively�relative�'relationships�%relationship�relation�relating�related relate   relaresistive�sensorc!schieppati�schwabscheme�sensitizek'sensitization	=sensitive
vsensing�senses�sensed�
sense�
senna:   �% ����jP< ����i@(����~bD�����_B����wbH$
�
�
�
�
p
O
3
	�	�	�	�	�	l	N	(	����}bH0����vaF*����sQ7����oK1
����fN2�����rO0�����c? ����]?%                             �n cisesNNPnsubjhelp�m confiningVBGpcompover"�l )conservativelyRBadvmodployed�k takingVBGpcompfrom�j highlyRBadvmodtrained�i #speculativeJJamodlifters!�h !creasinglyRBadvmodimportant�g liftersNNSnsubjachieve�f noviceNNcompoundlifters!�e 'investigatingVBGaclresearch�d difficultJJccompmake�c ationallyRBadvmoddefined�b operNNattrwas�a poorJJamodreporting�` #definitionsNNSnsubjmake�_ #involuntaryJJamodend�^ causingVBGadvclovercome�] demandsNNSdobjovercome �\ %overcomeVBconjcharacterize�[ inabilityNNpobjas�Z %characterizeVBPconjfeels�Y performVBccompfeels�X feelsVBZrelclpoint�W !volitionalJJamodly�V liftNNdobjcomplete�U downRPprtbreaks�T breaksVBZrelclpoint�S ersNNSpobjabove�R nitionsNNSnsubjconsider�Q defiNNPcompoundnitions �P !sensusNNcompounddefinition�O elucidateVBadvclrequired�N !continuousJJamodvariable$�M #'finegrainedJJamodconsideration�L begunVBNROOTbegun!�K #proximitiesNNSdobjdetermine�J stepNNcompoundfunction�I )thresholdbasedJJamodstep�H linearJJamodfunction�G purelyRBadvmodlinear�F exactJJamodnature�E fashionNNpobjin�D limitingVBGaclstudy�C lacksVBZadvclhave�B detrimenNNPcompoundeffect!�A )implementationNNdobjimpacts�@ boutNNnsubjimpacts�? %recuperativeJJamodperiod�> sitatingVBGadvcldecline�= necesNNPpobjwith�< advancingVBGamodneces�; declineVBxcomptends�: pacityNNnsubjtends�9 caMDauxtends�8 lossNNconjposition!�7 #precautionsNNSnsubjpasscome�6 stepupJJcompoundexercises �5 #intensitiesNNSdobjtolerate�4 tolerateVBccompis�3 !predisposeVBccompis�2 engageVBconjare�1 mandingJJacompare�0 deadliftsNNSpobjas�/ compoundNNnmodmovements"�. 'prescriptionsNNSdobjdeciding�- decidingVBGadvclis�, ationNNattris�+ !laseviciusNNcompoundal�* scantJJamodal#�) 'highthresholdNNPcompoundunits�( cruitNNPcompoundunits�' erloadNNdobjusing�& sightNNpobjin�% )validJJamodconfigurations�$ %ecologicallyRBadvmodvalid�# seekVBccompemploy�" #selectivelyRBadvmodemploy�! choiceNNattrbe�  !indicatorsNNSpobjas� manceNNcompounddecreases� perforNNcompoundmance� sustainedVBNamodmance� markerNNpobjto� spectVBpcompwith'� )musclebuildingVBGcompoundcapacity� markersNNSdobjinduces!� #continuallyRBadvmodtraining� impairVBrelclload� turnNNpobjin� promiseNNcompoundload� trainsVBZrelcldesigns� statedVBNamodmethods� ficultNNPattrare� !comparisonNNpobjfor� activityNNpobjof!� #controllingVBGadvclisolated� gramsNNSdobjisolated!� enduranceNNcompoundtraining� bicNNPcompoundtraining� aeroNNPcompoundtraining�
 rentNNcompoundtraining�	 concurVBadvclincluded� 'discrepanciesNNSpobjto#� !nonfailureNNcompoundprotocols� promotedVBDccompfound� vieiraNNPcompoundal� eficialJJamodeffects� benNNPcompoundeffects� negatedVBNconjindicated� prioriFWamodcriteria�  meetVBconjfound� ativeJJamodeffect�~ negJJamodeffect&�} +#sistancetrainedJJamodindividuals�| meetingVBGaclstudies�{ mizeNNcompoundincreases�z !maxiVBxcompobligatory�y !obligatoryJJacompis�x disputeVBPadvclenhances�w versialJJadvmodcontro�v controVBPxcompremains�u ureNNpobjto�t failVBcompoundure�s closerJJRoprdmade!�r #predictionsNNSnsubjpassmade�q %underpredictVBxcomptend�p peopleNNSnsubjtend�o expendedVBNacleffort   } ���_B+�����b@ ���uT2����x[@#����\@
�
�
�
�
n
R
5
	�	�	�	�	�	e	A	!	����sT3����z[< ����`@'����`K-����t[< ����iJ3����zT3����}V3                     �k devicesNNSdobjusing �j 'eccentriconlyRBadvmodtrain$�i )!concentriconlyRBoprdconsidered�h bandedVBNamodbiceps�g permittedVBNconjrecruited�f %femaleJJamodparticipants�e recruitedVBDconjwas�d merriganNNpobjto�c precededVBDadvclperformed�b haustionNNPcompoundgroup#�a #nontrainingNNPcompoundcontrol"�` )recreationallyRBadvmodactive�_ trindadeNNPpobjof�^ xweekNNPattrwas�] tinuedJJamodxweek�\ exertionNNpobjof�[ downsNNSpobjof�Z finkVBPROOTfink�Y mriNNdobjaging�X agingVBGrelclresonance �W %daysweekJJamodmeasurements�V !descendingVBGacldesign�U traditionNNpobjbetween�T !successiveJJamodweeks�S !nNNnsubjperforming�R tionalJJamodstrength�Q tradiNNnmodstrength�P dividedVBNccomptrained�O lowingVBGamodfol�N folNNnsubjpassdivided�M %maleJJamodparticipants�L weakerJJRamodfatigue�K !supposedlyRBadvmodweaker�J smallerJJRamodfatigue�I posedlyRBadvmodstronger�H supNNnpadvmodstronger�G linkNNattris �F !exhaustionNNnsubjpassbuilt�E equationsNNSpobjof�D !predictionNNconjtesting'�C +%bodycompositionJJamodimprovements!�B !preexhausVBPccompconsidered�A callyRBadvmodvalid�@ ecologiNNPpobjof�? !popularityNNpobjdespite�> !apparentJJamodpopularity�= #ditionNNapposrepetitions�< occurringVBGaclfatigue�; nificantJJcompoundfatigue�: sigNNcompoundfatigue�9 pauseNNdobjcomplete�8 occurredVBDconjrequired�7 setendNNPcompoundpoint�6 loweringVBGpcompbefore�5 secNNPnmodhold �4 traineeNNnsubjpassrequired�3 helpedVBDadvclperformed�2 assistantNNnsubjhelped�1 #heavierloadNNcompoundmvc�0 malesNNSnsubjperformed�/ #dropsetVBconjlighterload�. #lighterloadNNPconjered�- heaviNNpobjto�, eredVBDccompconsider�+ considVBDccompconsider�* !tisticallyRBadvmodsta�) #staNNPcompoundsignificant�( #preexerciseNNpobjto�' !reductionsNNSdobjshowed!�& !ciableNNPcompoundreductions�% appreNNPcompoundciable�$ postNNcompoundexercise�# #immediatelyRBpcompduring�" decrementNNpobjas�! etitionsNNSpobjof�  repNNcompoundetitions� ahtiainenNNPcompoundal� driversNNSpobjwithin� !indicativeJJacompbe� aptationsNNSpobjin � !temicNNPcompoundelevations � #accentuatedVBNamodexercise� dropVBconjforced� nalJJamodresponse� hormoJJnmodresponse� persetNNPcompoundtraining� suNNPcompoundperset&� /agonistantagonistNNPamodtraining%� 'preexhaustionNNcompoundtraining� titionsNNSdobjforced� supersetVBNccompreported� )agonistagonistNNpobjfor� niquesNNSpobjto� techNNcompoundniques� devoteVBxcompseems� experiNNcompoundence� gestedVBNccompused�
 #respondentsNNSpobjof�	 practicNNPcompoundes� surveyNNnsubjreported� advocatedVBNccompreferred"� #anecdotallyRBadvmodadvocated� negativesNNSconjsupersets� supersetsNNSdobjdropsets*� /prepostexhaustionNNcompoundsupersets� dropsetsNNSxcompforced� forcedVBNconjintended�  !themselvesPRPnsubjinclude� %exaggeratingVBGpcompby�~ #hanceNNPnmodadaptations�} !cializedVBNamodtechniques�| discussesVBZconjproposed�{ advancedJJamodmethods�z elsNNSdobjinvolving�y levNNcompoundels�x involvingVBGaclphase�w taperingNNamodphase�v peakingVBGamodphase�u optionNNattrbe�t coveryNNamodtraining�s sparinglyRBadvmodemploy�r %recuperationNNpobjon�q %consequencesNNSdobjreduce�p reduceVBconjmanage$�o +stimulusfatigueNNcompoundratio
�	#^TK=�.��s$������������}pd[P/B5*�c�������������|neZND8)���������Y����pbXD9$������������vh[�Q~C8."=
���M���������znjd�VK�69/#������R�������x)l^L;,!�
�
�
�#

�sg
�
�
�
�tg�
�
�
w
l
^
R
@
, 
 

	�	�	�	�w�	�	�	�	�	�	�	y	n	^	P	>	,		WK=1"�����������wk��^SD9*	�������D	���������ym^P@.         speeds�shearing�!slowtwitch�smooth�simulateespikes`
spike\
sleepWstopped	�stitutes�seriouslyB+stimulussensing+stimulusfatigueostimulus�stimuli	#stimulation�#stimulating	D!stimulated	stimulate�
still�stiffness�
stiffFsteroid2stepup6stepped�	stepJ
stemsw	stemgsteele�status�!statistics�'statistically�#statisticalstating�statementustated
state�stasinakiWstartsstarting
�stands �!standpoint �splicesspliced%similaritiesserving�+signalregulated�sequence�serial�sounds;%spectrometry:'sophisticated*	spot	speg
	soon�
serum�species�sorenessZstopping@#spangenburg)!shortening$!sequesters separated�sequester�
should>shortterm	V3shortermusclelength'shortermuscleshorter=shorten�%shortcomings�
shortshifts1
shift�shibata�sheathed�
shear�sex�severe%several�	sets �setend�set�-sessiontosession�7sessionsexercisessets�sessions
�session
Wserves|
serve
�serious�seriou�series�	seri�separates�separate
seo�	sent�sensusPsoleus��stabilityh!sternlichtf'stabilizationM	spanDsometimes7sternal.states
splitsmilios
spaces�    strengthhypertrophy/-strshepstone�%strengthened
q!strengthen	�strength �streamingBstrcng�!stratified
pstrategy6!strategies
Ystrains�strain�+straightforward|stores}stored�storagez	stor�
stops�standing�standards%standardized�%standardisedstandard�!standalone
�
stand�stance �!stagnationVstages �
stackstable	-#stabilizers�!stabilizer�sta�squatting�squats�
squatrsprinz�sprint�spread
�)sportsmovement!%sportsdiscus�sports�sportingz
sportYsponsessponse	splitting�splice�
spite
spine�spinae�
spent
speed�#speculativei!speculated	�speculate
�spectrum	�
spectspecifics�#specificity�%specificallyspecific �specif	�#specialized�)specialization
�	specspeaking
[spespatiallysparinglys
spans$
space�soy	k#southampton�source�
sound
�soughtL	sort�%sorptiometry
z
sonalRsomewhat	�	some �solent�solely�	sole^	soft!sociations�sociated	�social�socalled	�so�snp	�smpL
smithv	smds�smd�smaller�
small�slower[	slow2
slope�slightly�	slca_skinned�skinfold�skimmed	fskills�skeleton�skeletal
skele�	skel�sk9
sizes�sizesixG!situationsfsition_
sites	site�sitating>+sistancetrained�
sions�)siondetermined	X	sion�singlenu	�#singlejoint9single
@
sincesin@!simulationsimplyesimplesimilarlysimilarG	simi
0simy'significantly�#significant�signalsgsignaling
gsignal	�	signsights	�
sight&sig�sideringgsidered
B
siderH	sideh
shows	shownshowing
showed�	show�situation�
slack�shuttleyh   � ����iI&����mV7����t_K.�����eI.�����bH-
�
�
�
�
v
W
;
	�	�	�	�	�	p	U	5	�����`=�����uW3����tXC+����xX?# ����kG,����a8�����rY=����rP7                �k timeeffiNNnmodapproach�j noveltyNNpobjfor�i #implementedVBNadvclattain�h #lesserJJRamodadaptations�g attainVBPadvcliden�f vancedJJcompoundmethods�e odsNNScompoundeg�d tifiedJJadvcliden�c idenVBNROOTiden!�b #!discernibleJJamoddifference�a !occasionalJJamoduse�` equipVBPamodment�_ maticJJamodprag�^ pragJJccenhanced�] formingVBGpcompper�\ enhancedVBNpcompto�[ necessityNNdobjsupport�Z enhancingVBGpcompfor �Y #empiricallyRBadvmodsupport&�X =overreachingovertrainingNNpobjto�W prolongedVBNamodperiods�V !stagnationNNconjenjoyment�U enjoymentNNconjeffort �T !!perceptualJJconjsubjective�S !subjectiveJJamodresponses#�R !#representsVBZrelclrepetitions�Q arguaNNcompoundbly!�P wholebodyNNcompoundroutines�O dearthNNattris'�N )postexhaustionNNPcompoundtraining�M todateVBconjsuperset�L tricJJamodoverload�K eccenNNcompoundtric �J %overreachingVBGpcompbefore�I !regularityNNdobjcon�H siderVBdativecon�G !bloodbasedJJamodmarkers$�F )efforttrainingNNcompoundvolume�E 'incorporatingVBGpcompof�D %identifiableJJacompwere�C favorNNpobjin�B notableJJconjsmall�A nologyNNPdobjusing�@ idencedVBNaclev�? uedJJacompis�> continNNnpadvmodued%�= )#nonsignificantJJamoddifferences�< timescaleNNpobjby�; finiteJJamodtimescale�: ventionNNcompoundstudies�9 interJJamodstudies �8 #technicalJJamoddefinitions!�7 !!confoundedVBNROOTconfounded�6 reviewingVBGacleffort�5 deviceNNdobjused�4 mentsNNSpobjat�3 femurNNpobjof �2 #imalNNPcompoundrepetitions�1 fourCDnummodmuscles �0 %selectorizedVBNamodmachine�/ norrbrandNNPcompoundcom �. centuatedVBNadvclcontinues!�- !rewrappingVBGxcompcontinues�, continuesVBZadvclis�+ reachesVBZadvclputting�* puttingVBGaclcord�) cordNNdobjunravel�( unravelVBxcompserves"�' !flywheelJJcompoundtechnology�& pragmaticJJamodapproach�% !terventionNNpobjin�$ #welltrainedJJamodmen�# plateNNpobjof�" removalNNconjaddition�! releasersNNSdobjusing�  customNNcompoundweight� walkerNNPauxexplore� !efficacyNNdobjsupporting� existsVBZROOTexists� calNNPcompoundresearch� empiriNNPnpadvmodexists� %developmentsNNSdobjspite� spiteNNdepexists� achievesVBZccompclaim� !ufacturersNNSapposphase� rotatingVBGconjtilting� stackNNdobjtilting� tiltingVBGpcompby� releasedVBNaclmachines� xforceNNPcompoundmachines� %environmentsNNSpobjin� ginedVBNacompbeen� reimaNNPnsubjgined� mechanismNNpobjvia� pedalNNnmodmechanism� centricJJamodphase� assistVBconjperform�
 userNNnsubjperform�	 allowedVBDconjdeveloped� omniNNoprdcalled� nautilusJJamodomni� lineNNdobjdeveloped� jonesNNPnsubjdeveloped� arthurNNPcompoundjones� clesNNSconjincreases� vorNNpobjin� faNNPnmodvor�  #intermediusNNpobjfor� !actionNNdobjperforming#�~ #!essentiallyRBadvmodperforming�} resistedVBNadvclare�| !overloadedJJacompare%�{ '!eccentricallyRBadvmodoverloaded�z #isoinertialJJdobjusing�y therRBadvmodei�x eiNNPdobjusing�w explosiveJJamodsquats�v playersNNSdobjtrained!�u icehockeyNNPcompoundplayers�t horwarthNNPcompoundal!�s inertialNNPcompoundtraining �r #cepsNNPcompoundhypertrophy�q #biNNcompoundhypertrophy�p farthingNNPcompoundal�o triconlyNNPpobjbetween�n concenNNPcompoundtriconly�m crosssecNNPnmodarea�l higbieNNPcompoundal    ����vY=%
����jE+�����_>"
����bH(����uY: 
�
�
�
�
t
]
>

	�	�	�	�	m	T	7		����aH"����fL,�����aB����pY ���X4����yfC&����eE&����t\=                     �j 'onymousJJamodperiodization�i 'synNNnmodperiodization�h illnessNNpobjof�g #occurrencesNNSconjfewer �f formanceNNcompoundplateaus�e fewerJJRpobjwith!�d #assumptionsNNSpobjregarding�c debateNNattris�b csapoNNPconjfisher�a !periodizedVBNpcompfrom�` #contractileJJamodtissue�_ mesocycleNNpobjwithin�^ ditionsNNSpobjbetween�] teristicsNNSpobjthan�\ characNNcompoundteristics!�[ #acteristicsNNSconjvariation�Z #charNNcompoundacteristics�Y explainVBROOTexplain �X %betweenstudyJJamodvariance�W τNNpunctp�V qLSROOTq�U %transferenceNNnsubjis�T !ariseVBPconjcontribute�S questionsNNSnsubjarise�R !macrocycleNNpobjin�Q !mesocyclesNNSpobjof�P hopeNNpobjwith!�O !ostensiblyRBadvmodenhancing$�N %!successivelyRBadvmodpotentiate �M %subsequentlyRBadvmodintend�L periNNPcompoundods �K !organizingVBGxcompinvolves!�J 'nameNNcompoundperiodization�I improvingVBGpcompof6�H 77planningperiodizationNNROOTplanningperiodization�G cardioNNpobjto�F !infeasibleJJacompis�E apartRBadvmodschedule�D boutsNNSdobjschedule�C scheduleVBxcompseem �B !mediatorsNNSconjmoderators�A sourceNNattrare!�@ +vidualJJamodcharacteristics�? detrimentNNpobjwithout�> overlyRBadvmodexcessive�= #assumingVBGaclperspective�< #perspectiveNNpobjfrom�; sionsNNSdobjdraw�: concluNNcompoundsions�9 absentJJacompwas�8 ogeneityNNnsubjwas�7 heterNNcompoundogeneity�6 revealVBconjfound�5 ticipantsNNSpobjof�4 treatmentNNcompoundeffect�3 schumannNNPcompoundiden�2 thereofRBadvmodlack�1 %insightNNdobjmetaanalyses�0 offersNNScompoundinsight#�/ ##contrastingVBGamodconclusions�. ualJJamodstudies�- !historicalJJamodvariation�, plainRBadvmodex �+ #nutritionalJJamodpractices�* isticsNNPcompoundtraining"�) characterNNcompoundpractices�( alongsideINprepbe�' culpritNNattrbe�& intensiNNPcompoundty�% overlapNNnsubjbe�$ #plificationNNattris�# #oversimJJamodplification�" adaptiveJJamodre�! !exceedingVBGaclmodalities�  cyclingVBGconjrunning� runningNNpcompwith� impactedVBDROOTimpacted� anceNNcompoundtraining� endurNNPpobjof� sampleNNcompoundsize� adjustedVBNconjreported� !deviationsNNSpobjby� !pretestJJamoddeviations� deltaNNdobjusing� nationNNdobjusing� combiJJamodnation� !unweightedJJamodnation� wilsonNNPcompoundal� tenuatedJJpobjat� respectNNpobjwith� aptationNNROOTaptation � 'corroboratingVBGaclstudies� emergedVBNconjinhibit� 'demonstratingVBGaclwork� !downstreamJJamodtargets� rapamycinNNpobjof�
 mammalianJJamodtarget�	 #inhibitVBconjparticipate� pathNNcompoundway#� 'monophosphateNNcompoundkinase&� 'adenosineNNcompoundmonophosphate� essenceNNpobjin� #explanationNNpobjas� !offeredVBNconjattenuated$� '#endurancetypeJJamodadaptations� volvedNNpobjin�  !attenuatedVBNccompis� %interferenceNNnmodeffect�~ concernsNNSattrbeen�} hicksonNNPpobjof�| classicJJamodstudy�{ decadesNNSpobjfor"�z %colloquiallyRBadvmodreferred�y dioNNPconjtraining�x carNNcompounddio&�w -aerobicenduranceRBadvmodtraining�v %concurrentlyRBadvmodalong�u !concurrentJJamodtraining�t aroundINprepforces�s doingVBGpcompin�r ensuNNPnmodmovement�q decreaseVBxcompserve�p fatiguingVBGadvclpresent�o careerNNpobjover�n moveNNcompoundments�m valsNNSconjsets�l cientJJamodapproach
	� � 	�R	�	��	�	�	�	�	�
�	�	m	^	J	=	-	#		����������~k\QB, 
����������y_UIB6*����I��������u}tiXJ<1"����������xcO�=(Se�
r4����
��
�
>
M�-��ymZK<2%
%�����������{p]g^
dPBE8. ��<���������ykbRJ*����
������ug
�XG:)� �������zjXG7*�����������thZ
�I<-
0
Y���������
�
�
��{
'�oulatedAuktypes�type3  ulatedAukpcpartpickercomlistcvxrt^uk�!ufacturersued?ualizing
�ual�ustyrosine-tyr2typicallyytypical�%typespecific
#
types�type3ty�two�%tweensubject�!tweengroup
�
tween
lturnover	c	turn	turewtuberous5tuationsttn�tsc7
trying
trunk�truncat
�
truly	true	�trophyZtrophic
�trolledCtrivial�trindade�'triglycerides�triggers	@trigger�triconly�triceps	tricLtributory�tributing	�'triangulation	�trials�
trial	n!tremendous@treatment�!traversing;#transportera#transmitted�transmit%transmission%translatable	Ktransient�%transference�+transferability%transduction�)transcriptomic	�+transcriptionalU'transcriptionY3transcribedimported�tralized/trainstrainingtrainees@trainee�trained�
train'traditionally�#traditional
-tradition�
tradi�tradeoff?tractile�#tracellular
f
trace)tra
.towards�totality@
totalStostudy�	toryL	torsntorqueo!torgluteus~
topic`
tonal~tonJtolerate4tolerable
�together�todateMtocols
{totively	�	tive�5titrationprogression�titlesItitions�tition='titinencoding�
titin!tistically�tissuesYtissue`tionship�
tions�tional�	tion�tinued�tinelyJ-timeundertension�timescale<
timestimeframe
�'timeefficient
Ftimeeffik	timetilting
tight�tifiedd!tification
�
tificGtidybayes�
ticks�ticipants�ticipantntic�tibialiso	thus �!throughoutthrough)thresholdbasedIthreshold
7-threedimensionalg
threethoughtOthoughU
those	this
third�
thigh�thickness�	they =theses]	thesejthereof�thereforep
there �	upsnunstable_trapezius'trauma
untilttubules�#translation�unloading�transportm)ultrastructurekthreath#upregulatesP!upregulateD!transports;unbound4#upregulated� thereafterundergo�treadmill�twofoldg	tookR%transgenesis,!transgenic+toward+traininginduced!transducedtriple� theorized�thighs�upstream�too�'transplantingkuninjuredh   therebya%trainability]triggered=#translocatetubulin
thick%tryptophanes�uptake�upright�%upregulationLupperlimb:upperbody�
upper�	upon�updatedupdate�up �!unweighted�unwanted5untrained
�%unresponsive
|unravelediunravel(#unpublished"unlikely@unless�unknowni!university�#universally �universalr
units�	unit(unique!unintended1!uninformed	unin	!unilateral�uniformx%unidentified	�unfolds�!undulating#undoubtedly
�%undetermined�!undertaken2undertake	*!understood�'understandingB!understandH%underpredict�%underpowered�underpin	�undergone)underdeveloped
�underde
�
under#uncontestedBunclear�#uncertainty�uncertain�unbiased�#unavailable�!unanswered�unable&un�ume
�	ulum�+ultrastructural!ultrasound
u!ultimatelyNultimate	   ~& ���~\;�����oT;�����sX6����vW6����yc=
�
�
�
�
|
[
;
!
	�	�	�	�	w	U	2	�����jK+����z^<����oU8�����vX?����vU=�����cM,����tT<����^C&                                  �h !candidatesNNSattrare�g signalsNNSnsubjare%�f -stressassociatedVBNamodtriggers)�e 9damageinjuryassociatedVBDaclstimuli�d discussVBPadvcldefine�c sensorNNconjstimulus�b elusiveJJacompremained-�a 9hypertrophystimulatingNNcompoundstimuli�` cascadeNNpobjof�_ causesVBZrelclmechanism�^ soleJJamodmechanism!�] 'mtorcmediatedJJamodincrease�\ #strikingJJamodadaptations�[ exerkineNNPadvclissued�Z issuedVBNaclcanadian�Y canadianNNdobjhas�X patentNNcompoundcanadian�W hePRPnsubjhas�V submittedVBNamodwork�U nancialJJamodsupport�T nonfiNNPnmodsupport�S feesNNSpobjper�R sonalJJamodfees�Q conductNNpobjduring�P councilNNpobjfrom�O dairyNNcompoundcouncil�N grantsNNSpobjof�M reportsNNScompoundgrants�L smpNNPcompoundreports�K )fitnessrelatedJJamodsmp�J #tonNNPcompoundcorporation$�I +outliningVBGaclrecommendations�H condensedJJamodsummary �G %gramNNcompoundprescription�F fuNNcompoundture�E encourageVBPrelcldata�D firmJJacompare�C trolledVBNamodstudy�B makNNPnsubjing�A cusedVBNadvclutilized�@ foNNPnsubjcused�? #amelioratedVBNadvcllong�> optimizedVBNROOToptimized�= %organizationNNpobjof�< dailyJJconjweekly$�; #'alternatingVBGaclperiodization�: periodizeVBadvclbe�9 deloadsNNSconjvolume$�8 %!interspersedVBNadvclperiodized"�7 )highrepetitionNNcompoundsets�6 demandingVBGamodexercises�5 !pairingVBGadvclundertaken�4 hensiveJJamodmanner�3 compreNNPcompoundmanner�2 !undertakenVBNadvclis�1 odizedNNcompoundapproach�0 !illustrateVBxcompis�/ strictlyRBadvmodis�. possiblyRBadvmoddif�- !weektoweekNNPpobjwithin�, antretterNNPcompoundal�+ organizedVBNpcompthan�* ensuringVBGaclweeks�) !terspersedJJpobjin�( santosNNPnsubjreported�' dosVBZconjfavor�& narrowerJJRamodrange�% broaderJJRamodspectrum �$ #examinationNNdobjrequiring�# #happenVBPrelclperformance�" hapsNNSpobjper�! minorityNNnsubjexamine�  #flictFWapposconclusions� fectsNNPdobjconcluded� !undulatingVBGpcompto� atedVBNacldeline� delineJJnsubjvary� phasesNNScompounddeline� !phasicallyRBadvmodganize� !ganizeVBconjincreasing� principalJJamodgoal� %inconclusiveJJacompis&� 'lowrepetitionNNPcompoundtraining� !etitionNNPdobjperforming� highrepNNPnmodetition#� 'predominantlyRBadvmodcomposed� endsNNSpobjat� #hypothesizeVBPccompseems!� %injuriesNNSdobjpowerlifting� winwoodNNPconjkeogh� keoghNNPpobjof� pulldownsNNSconjpresses"� !%consistingVBGaclloadspectrum� %loadspectrumNNpobjacross�
 stressesNNSnsubjdiffer�	 grossJJamodoutcomes� pendentNNdobjoccurring� indeJJamodpendent� !marilyRBadvmodinfluenced� priPRPnsubjdiffer� plyingVBGpcompap� seasonNNpobjto� !programmedVBNccomprequire� overtrainJJamoding�  accrualJJpobjfor� planVBpcompto�~ tationNNdobjforecast�} adapNNPcompoundtation�| #requirementNNattris�{ needingVBGpcompwithout �z %applyingVBGaclmanipulation�y syndromeNNpobjof�x onsetNNdobjprevent�w odizationNNnsubjis�v bufordNNPcompoundal�u dizationNNdobjadopting�t perioNNPpobjof�s steppedVBNccomprequire�r continueVBPadvclstepped�q debatesNNSnsubjcontinue�p forecastVBNxcomprequires�o 'predeterminedVBNamodtimes�n !successfulJJamodresults�m variNNcompoundation#�l 'nonperiodizedNNPnsubjtraining�k plannedVBNamodvariation   �� �����hSF3�������p`B,�����o^Q'��������tYA0!�������sTA%
�
�
�
�
�
�
}
\
A
$
 	�	�	�	�	�	~	m	Z	F	+		������xiZG7 �����kXB+������zfQ1������mT<*��                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    �C +shepstone et alORG�B 'derecruitmentPERSON�A !schieppatiORG�@ nardoneORG�? igf mrnaFAC�> 'three secondsTIME�= )the fifth weekDATE�< -the first minuteTIME�; 9less moderate  secondsTIME�: !veragarciaPERSON�9 behmORG�8 andersonPERSON�7 dozensCARDINAL�6 -fasttwitch fiberORG�5 5longer than one hourTIME�4 7those with plus yearsDATE�3 #rm  minutesGPE�2 fourthORDINAL�1 #multiplesetORG�0 /bodybuildingstyleORG�/ #hypertophicPERSON�. #microtraumaGPE�- pradoGPE�, !slowtwitchNORP�+ !fasttwitchORG�* 1calcineurin camkiiORG�) %after  weeksDATE�( maimum rmPERSON�' kubotaPERSON�& #three gramsQUANTITY�% 3aquaporin aquaporinORG�$ 7cellular hydration ieORG�# myotraumaGPE�" halflivesORG�! 1autocrineparacrineGPE�  leydigORG
� mgfORG� 'igfeb  levelsORG� a dayDATE� igfecPRODUCT� igfebORG� igfeaORG� -igf testosteroneORG
� jnkORG� +cjun nhterminalORG� p mapkORG� %aktmammalianNORP� calciumcaPERSON� mapkORG� )rapamycin mtorORG� mrfPERSON� /myf myod myogeninORG� !myonuclearORG� myonucleiGPE� mrnaPERSON� !myonucleusPERSON� #mitoticallyPERSON�
 kelleyPERSON�	 )noncontractilePRODUCT� morganPERSON� lynnPERSON!� Amuscle crosssectional areaGPE� 1a couple of monthsDATE� )matt alexanderPERSON� 5jari ylänne ju chenPERSON� octoberDATE
� jjhORG�  zdisksNORP� 5musclespecific forceORG�~ )musclespecificPERSON�} #inoki et alPERSON
�| hekORG�{ 'membraneboundPERSON�z +george brooks  PERSON�y mmolkgPERSON�x 1mmolkg preexercisePERSON�w teschNORP�v lohmannORG�u 7pcr resynthesizes adpPERSON�t %within  daysDATE�s rosPERSON�r ilPERSON�q eimdORG�p monthsDATE�o %nearlifelongGPE�n eimd
ORG�m cooccurORG�l )hubal  proposeORG�k )frey et al  toPERSON�j yap  PERSON�i ppxyPERSON�h casaORG�g nonmuscleGPE
�f tscORG�e filamincPERSON�d halfCARDINAL
�c ttnORG�b %several daysDATE�a #up to  daysDATE'�` Gcostamerebased mechanosensorsPERSON
�_ pskORG�^ mtorPERSON$�] Gdiacylglycerol kinaseξ dgkξORG&�\ Kzdisclinked phospholipase d pldORG�[ al  firstPERSON�Z mtorcPERSON�Y latPERSON�X taz yapPERSON�W ankrdNORP�V ptenPERSON�U mtorc yapPERSON�T /coactivating teadPERSON�S tazPERSON�R wwtr  yapORG�Q yapPERSON
�P fedORG�O tsc mtorORG�N 9dystrophinglycoproteinGPE�M zdiskNORP�L )vinculin talinPERSON�K )mechanosensorsORG�J /alfred goldberg  PERSON�I thirdORDINAL
�H igfORG�G #protein·kgGPE �F ?diseasespecific mortalityORG�E allcauseORG�D )half a millionCARDINAL�C )mechanosensingPERSON�B #mtorc hippoPERSON�A canadianNORP   } ���mG����oT9����iJ,����mQ:�����cH(	
�
�
�
�
l
J
'
	�	�	�	�	o	Q	<	�����hL.�����fJ'����qP,����aC�����}R/�����jP3�����{aB'����oP*      �e #shearVBcompoundcompression#�d )mechanosensorsNNSapposstimuli�c detectVBPrelclplethora�b withstandVBrelclmuscles �a %cytoskeletonNNconjskeleton�` skeletonNNapposmuscles�_ !structuresNNSdobjevolved�^ beingsNNSnsubjevolved�] wonderNNattris�\ organismsNNSpobjof�[ loadedVBNamodorganisms�Z msNNnmodorganisms�Y gravityNNnsubjis�X #environmentNNpobjin�W evolvedVBDROOTevolved�V earthNNpobjon�U lifeNNnsubjevolved�T sensingVBGpcompof�S capableJJacompare�R causedVBDccompshowed�Q oppositeJJamodleg�P wkNNPnsubjusing�O exercisedVBDccompis$�N %workloadNNcompoundparticipants�M caveatNNpobjof�L syntheticJJamodrate�K !fractionalJJamodrate�J labeledVBDcsubjis �I extensorNNcompoundexercise(�H -timeundertensionNNPcompoundproduct�G yrNNconjyoung�F   _SPdep±�E ±NNPpunctyoung�D  _SPdepyoung�C nearJJamodresponse�B #addressVBadvcldistinguish,�A 3#hypertrophyinducingVBGxcompdistinguish�@ #distinguishVBPadvclis�? linksVBZrelclstimulus�> !metabolismNNpobjas�= hostNNnsubjis�< issueNNpobjin$�; 'hypertrophiedVBDadvclconcluded�: stretchNNconjablation�9 #castinducedJJamodstretch!�8 !flexorNNPcompoundsynergists�7 plantarNNPcompoundflexor�6 plantarisNNpobjas$�5 %!mechanicallyRBadvmodoverloaded�4 goldbergNNPnsubjhave�3 alfredNNPcompoundgoldberg�2 normalJJamodpattern�1 refNNcompoundthis �0 )immobilizationNNcompoundeg�/ atrophyNNapposlines�. intuitiveJJamodstimuli�- updateNNdobjprovide�, !directionsNNSpobjof�+ whereverWRBadvmodpossible�* reconcileVBxcompaim�) thirdJJconjsecond�( actVBaclevidence�' #identifyingVBGpcompof�& actualJJamodstimuli�% matterRBadvmodare�$ whyWRBadvmodare�# alterVBPrelclevents�" eventsNNSdobjsignaling�! !expressionNNnsubjchange�  spliceNNcompoundvariant$� 'mechanogrowthNNPcompoundfactor� igfNNPpobjas� !regulatorsNNSnsubjare� sensesVBZrelclsensor� #firstinlineNNPpobjas� sensorsNNSnsubjare� %transductionNNdobjtrigger� triggerVBPrelclstimuli� !unansweredJJacompremained � !moleculesNNSdobjidentified� !understoodVBNconjexercise� regulateVBPconjexercise� genesNNSconjpathways � ablationNNcompoundpathways!� synergistNNcompoundablation� rodentsNNSpobjin� reducesVBZconjprevents� blockadeNNnsubjprevents� hubNNdobjsignaling� 'wt−·day−-LRB-punctis� #protein·kgDTcompoundbody�
 containsVBZrelcldiet�	 dietNNdobjconsume� consumeVBconjperform� inbetweenNNPamodsets� !exercisersNNSnsubjperform� ≈NNPpobjwith� nutritionNNpobjwith� diseasesNNSpobjfor� incidenceNNnsubjis � diseaseNNcompoundincidence�  mortalityNNconjallcause(� +diseasespecificNNcompoundmortality�~ allcauseNNpobjwith�} millionCDnummodpeople�| halfPDTquantmodmillion�{ longevityNNconjhealth�z sportingVBGpcompfor$�y !!metaboliteNNcompoundcandidates�x #metabolitesNNSnsubjbe)�w /#exerciseregulatedNNPamodmetabolites"�v 'combineVBPrelclinterventions�u probablyRBadvmodis�t adhesionsNNSpobjof�s focalJJamodadhesions�r #equivalentsNNSattrare�q !costameresNNPpobjto)�p 5deformationinitiatedJJamodsignaling'�o 5nuclearJJamoddeformationinitiated(�n )!mechanosensingNNcompoundmechanisms#�m !candidateNNcompoundmechanisms'�l %'incompletelyRBadvmodcharacterized"�k autophagyRBcompoundsignaling�j hippoNNPpobjof*�i 5!filamincbagdependentJJamodregulation
% � �)�������ulbXL7&�����������qd[I=+��������ti`TJA6'
����������yn]S=6)	���������wmaSC2'
�
�
�
�
�
�
�
�
�
�
�
w
d
U
D
3
!

		�	�	�	�	�	�	�	�	�	�	x	l	Z	P	D	,			����������yl]N?2%����������wlbRE7&��������w`M9#��������qbSE6*����������6teUL;)��������xlXB1$�����������{ocYLB              #luteinizing� linnamo�merits�	milo�million}	milk	gmilieu�milder
,migrate�
might�midway�midfootxmiddlemen�middle�mid�#microtrauma�#microscopic@
microW	mice�mhmgfmetricvmethods#methodology	2-methodologically	()methodological�method�	meth�
meter�)metaregression�metallic�#metabolitesx!metabolitey!metabolism�metabolicH-metaanalytically>%metaanalytic
�%metaanalysis?%metaanalyses�#metaanalyse�met!mesocycles�mesocycle�merrigan�
ments4mentioned 	ment�men�membranes�'membranebound�membrane�melbourne�meeting�	meet !mediumterm	�+mediumintensity�medium�mediators�mediator�mediation�mediating�mediates�mediated�mediate5median�medialis�medial
media�3mechanotransduction61mechanostimulation�)mechanosensors�'mechanosensor-mechanosensitiveT)mechanosensingn'mechanogrowth�/mechanochemically�mechano+mechanistically#mechanistic�!mechanismsCmechanism%mechanically�!mechanical�mechanameasuringcmeasures�%measurementsz#measurementwmeasured�measurev!measurableR	measT
meantL
means%meaningfully!meaningfulEmeaningd	mean�#mealinduced	I	meal	#me�mcmaster�mcmahonHmckendrymcbride*mcfmay �maximus*maximum�!maximizing+maximizes�maximizedpmaximize%maximizationqmaximise0maximallyemaximal�	maxi�max	�matters\matter�	matt\matrix�
matic_materials�material�matched	�1master’sdoctoralmaster\	massq%martinezcavaimarkov%markersmarkermarked�marily!marginally�margin1marathon[	mapk�	many!!manuscriptm%manufacturer�manual �manner3%manipulation �%manipulatingQ#manipulated�manifests�'manifestationmanifest�manding1
mance!management�manage
�manmammalian�
males�	male�mal	�making	M
makes�	make �makBmajorityW
major)#maintaining!maintainedzmaintain�mainly%	main"maimum�magnitudemagnified=magnetic
r	maeoI	made�#macroscopicV#macrophages�)macromolecules�!macrocycle�machines�%machinebasedYmachine-m�-lysophosphatidic�	lynn�#lymphocytes�
lying�ly
�
lungewlumbar�
ltype"lowvolume'lowrepetitionlowloads
Llowload	�%lowintensity�lowing�lowest1lowerlimb7lowering�lowerbody�
lower�low	s	loss8losing;los
lopez
 looking5looked�longterm!
longo*-longmusclelengthF)longitudinally�%longitudinal�longevity{!longerterm	]1longermusclelength%longermuscle%longerlengthJlonger�%longduration^longClohmann�logically�logical
logic�log
loennekeI
locus	
locatedylocallylocalizedlocalize�
localC%loadspectrum
loads�loading�#loadinduced0+loadindependent	�loaded�	load�	lnrrllc�living�
liverlittle�!literature �
liter
listsz	listo	lish�
links�   linkingfl#microcycles�   y � ����aE#	����mU; ���~T/
���{Y7����dC%
�
�
�
�
�
a
A
 
		�	�	�	�	z	a	C	!	��}^4����hT4����wH$	����uS/����qL,����vP4�����mE����X8 � �^ ptenNNPdobjsuppress�] inhibitorNNPdobjsuppress�\ suppressVBxcompreported%�[ 1exerciseassociatedJJamodstimuli#�Z +differentiationNNdobjregulate$�Y 'transcriptionNNcompoundfactors�X teadNNpobjby�W %coactivatingVBGamodtead�V cofactorsNNSattrare$�U +transcriptionalJJamodcofactors%�T -mechanosensitiveJJamodcofactors�S wwtrNNPcompoundyap�R tazNNPcompoundgene�Q paralogueNNPcompoundgene�P 'yesassociatedVBNamodgene�O yapNNnmodgene�N effectorsNNSdobjactivated�M plcγNNPpobjby�L cγNNPcompoundplcγ#�K 'phospholipaseNNPcompoundplcγ�J pipNNPpobjof �I %bisphosphateNNPcompoundpip(�H 5phosphatidylinositolNNPcompoundpip�G !conversionNNdobjpromote�F stiffJJconjsoft�E !attachmentNNpobjof�D )phospholipasesNNSpobjas"�C )acidgeneratingVBGamodenzymes�B %phosphatidicJJamodenzymes�A #contributesVBZccompis!�@ 'posteccentricJJamodexercise�? )phosphorylatedJJamodtyr�> fedVBDconjfasted�= #fastedVBNamodindividuals!�< +phosphorylationNNPcompoundh�; +activityrelatedVBNamodfak�: activatedVBNccompis�9 skNNPcompoundsignaling�8 #mtorNNPconjhypertrophy�7 tscNNPcompoundmtor�6 sclerosisNNcompoundtsc�5 tuberousJJamodmtor!�4 !#igfinducedJJamodhypertrophy,�3 3autophosphorylationNNnsubjpassrequired&�2 3tyrNNcompoundautophosphorylation�1 myotubesNNScompoundigf�0 ccNNPcompoundmyotubes�/ culturedJJamodigf�. movesVBZrelclkinase�- tyrosineNNcompoundkinase�, #nonreceptorJJamodkinase�+ ptkNNpobjby�* encodedVBNaclfak�) fakNNPnsubjis(�( 3costamereassociatedVBNamodproteins�' dystrophyNNpobjas"�& duchenneNNPcompounddystrophy�% severeJJamoddiseases�$ dmdJJamodgene'�# 1dystrophinencodingNNPcompoundgene�" mutationNNnsubjresults-�! 7vinculintalinintegrinNNPcompoundcomplex.�  9dystrophinglycoproteinNNPcompoundcomplex"� costamereNNcompoundcomplexes� transmitVBconjconnect� !zdiskNNnpadvmodassociated� exteriorNNdobjconnect� connectVBconjilk� substrateNNpobjon� ilkVBconjbecame � )integrinlinkedJJamodkinase� kinasesNNSconjtalin� integrinsNNSconjtalin� talinNNPpobjas� vinculinNNPcompoundtalin� complexesNNSpobjthrough� !anchorVBPccompdiscovered� onwardsRBadvmods� noncancerNNcompoundcells� anchorageNNpobjwithout� agarNNpobjon� softJJamodagar� !growVBccompdiscovered� cancerNNcompoundcells�
 !discoveredVBDadvclbecame�	 becameVBDROOTbecame$� %!historicallyRBadvmodmechanical+� -)costamererelatedVBNamodmechanosensors#� %transmissionNNcompoundsystems� #filamincbagNNPconjtitin� titinNNPdobjdiscuss� initiatesVBZrelclresponse� modifiesVBZrelclmechanism� 'mechanosensorNNpobjfor2�  7'hypertrophytriggeringNNcompoundmechanosensor� !equivalentJJamodadhesion�~ adhesionNNattrare"�} !membraneNNcompoundsarcolemma"�| #laterallyRBadvmodtransmitted'�{ )#longitudinallyRBadvmodtransmitted'�z -forcetransducingNNcompoundsystems�y bonesNNSconjtendons�x tendonsNNSpobjto#�w ##transmittedVBNROOTtransmitted�v striatedJJamodfibers�u examplesNNSattrare�t valuesNNSnsubjare�s µncellNNpobjof%�r #fibroblastsNNSnsubjpassreported$�q /actincytoskeletonNNpobjthrough�p pnUHnmodcells�o myosinNNcompoundhead�n µnNNdobjgenerate�m skinnedJJamodtype�l nonmuscleNNPcompoundcells�k generateVBPadvclare�j surroundsVBZrelclmatrix�i matrixNNpobjof �h #stiffnessNNconjcompression�g #compressionNNpobjas&�f ##deformationNNcompoundcompression   ~ ����_@����eE+����}bL&����mV+���~hL*
�
�
�
�
u
V
4
	�	�	�	�	�	o	Q	3	����{b@ ����y[= ����u^;#
����dM5����tW1�����qW8�����dG(����jQ1           �\ #actinlinkedJJamodfilamins�[ generatesVBZconjdiscuss�Z pairNNdobjdeform�Y deformVBaclfilamin�X !homodimersNNSdobjform!�W !vshapedNNcompoundhomodimers�V filaminNNattris*�U /actincrosslinkingNNcompoundmolecules�T filaminsNNSnsubjare�S effectorNNPcompoundyap�R !activatingVBGpcompof�Q localizeVBPccompcause�P flncVBPROOTflnc�O filamincNNPcompoundgenes�N exceptINprepis�M domainNNpobjwith!�L )murfproteasomeNNpobjthrough&�K %interactionsNNSnsubjpassreported�J unfoldsVBZconjincrease�I situationNNpobjin�H actuallyRBadvmoddecrease�G moleculeNNpobjwithin�F %consequentlyRBadvmodgo�E slackRBadvmodgo�D goVBROOTgo�C shortenVBconjgenerate�B actinNNconjmyosin#�A #actinmyosinNNcompoundproteins�@ liesVBZccompconsider�? #terminologyNNnmodload(�> +'exerciserelatedJJamodmechanosensor�= numerousJJamodproteins �< 'phosphorylateVBccompcauses�; bindingNNdobjbind�: bindVBccompallowing�9 atpVBNnsubjbind�8 pocketNNpobjof�7 !atpbindingVBGamodpocket�6 pullsVBZadvclactivated&�5 -stretchactivatedNNcompoundkinase �4 passivelyRBadvmodstretched�3 !elasticityNNpobjto�2 elasticJJacompis�1 portionNNpobjof �0 'ibandspanningJJamodportion�/ mlineNNPpobjto �. !myopathiesNNScompoundtitin�- 'titinencodingVBGamodgene�, mutationsNNSpobjas�+ giantJJamodprotein�* ttnNNPROOTttn�) strainNNdobjexerts�( exertsVBZccomplikely�' sensedVBNconjconnects�& connectsVBZrelclsites#�% )postresistanceVBPxcompprotein�$ %cytoskeletalJJamodloading�# dependsVBZconjrespond�" respondVBccompsuggest �! )differentiatedJJamodmuscle�  togetherRBadvmodprevent� !inhibitorsNNSconjintegrin� integrinNNpobjas� isotonicJJamodmedium� uptakeNNdobjincreases� glutamineNNcompounduptake� mediumNNcompoundincreases� #hypoosmoticJJamodculture� cultureNNpobjby� broughtVBNaclmyotubes� ratNNcompoundmyotubes� swellVBconjlast� %interstitiumNNpobjof� definiteJJamodevidence� edemaNNnsubjlast� eimdNNdobjswelling� !describedVBNaclperception� !perceptionNNpobjin� !temporaryJJamodperception� brieflyRBadvmoddiscuss� pumpNNpobjas)� ))costamerebasedVBNamodmechanosensors�
 activatesVBZconjactivated�	 residuesNNSpobjat� wildtypeJJcompoundmice� miceNNSnsubjhave� )overexpressingVBGamodmice� linkedVBNrelclgroup� itgaNNPpobjby� isoformNNnsubjpasslinked#� %αβintegrinNNcompoundisoform� taskNNattris.�  5'synthesisstimulatingVBGamodmechanosensor� )identificationNNnsubjis(�~ 5hypertrophymediatingNNconjstimulus�} dgkξNNPpobjby�| kinaseξNNPcompounddgkξ$�{ )diacylglycerolNNPcompounddgkξ�z !reactionNNdobjidentified�y locatedVBNconjis�x !generatingNNamodenzymes�w enzymeNNpobjas#�v -acidsynthesizingVBGamodenzyme�u pldNNPadvclis�t #zdisclinkedJJamodd�s youPRPnsubjpld�r butanolNNpobjby�q !inhibitionNNnsubjprevents�p anteriorJJamodmuscles�o tibialisNNPpobjin"�n 'concentrationNNdobjincreased�m regulatorNNattris�l modulatesVBZdepsuggests�k sensitizeVBrelclabundance�j abundanceNNdobjincrease�i unknownJJconjas�h scenarioNNdobjsuggests"�g %collectivelyRBadvmodsuggests�f !situationsNNSpobjin�e ankrdNNcompoundincreases�d #latencodingJJamodgenes!�c )hypertrophyingVBGamodmuscle"�b -synergistablatedJJamodmuscle�a #transporterNNPdobjencode�` !encodeVBPrelclexpression�_ slcaNNPpobjof
� � @*���xj\I4!���������th\N?1"����������udSA�/!��������|l[H9)
�
�
�
�
�
�
�
�
�
y
k
\
O
F
;�
$

	�	�Y	��	�	�	�	�	�	v	f	V�	H	8	'�hu		���������yncUF�5&��������zhN?.����������wl`WL�>2$������������raSC3%�����������sfXO@6*����������tgUJ@8.#�����������yndYI<.!hesejthereof )tensioninduced�	thesejthereof�thereforeptherebya!thereafter
there �	ther�
theoryctheorized�theories�#theoreticalR	theo�	then7!themselves�	them�theless	�theiusca�theirsv	theirlthethat<thanks�
thankW	than�th	�	text�
tests%testosterone	testing}testex�testes+tested�	test 
tesch�#terventions�!tervention%!terspersed)#ternatively�
terms
#terminology�!terminated
5terminatemtermed�	term �teristics�ter�tenuated�tensity
4tension�
tenet"
tends
Mtendons�tending�	tend~tenntempos=temporary�	tempo5templateT
temic�#tematicallyRtem
J	teinNtegrates	�!technology�!techniques technique#technically)technical8	tech�
tears�tearing�	team�	teadXtazRtaxing
Gtax�tation�%taskspecific
tasks&	task�targets �targeting
%targeted�targetktaperingw
talintal�takingk
taken	take6takarada~	tain�tachmentsd
table't�systems	�systemic)systematically�!systematic>system	�sys
Isynthetic�#synthesizes#synthesized)!synthesize#5synthesisstimulating�%synthesising�synthesisO
synpo�!synergists�+synergistically#synergistic	-synergistablatedbsynergist�synergis�syndrome�%synaptopodinsyn�symposiumQ#sympathetic�switching�swelling�
swell�
sweet
sways�sustainedsustain�suspends�#susceptiblesurvey�surrounds�!surrounded%surprisingly	!surprising	|surmisedxsurgery�#suppressionA!suppressed�suppress\!supposedly�!supportive�!supportingCsupported9support{supplied�+supplementation�'supplementary�!supervised�supersets�superset�#superiority�superior/supercompensation
�
super1sup�summaryWsummarize/suggests�!suggesting�suggestedqsuggest �sug�%sufficiently�!sufficientsufferssuch.%successively�!successive�%successfully}!successful�!substrates�substrate'substantially	�#substantialB!substances�%subsequently�!subsequent	m
subse�!suboptimal�submittedV!submaximal
=subjects�!subjectiveSsubjectedsubjectMsubgroups�subgroup%subdivisionsh%subcutaneous�#subcategory�#subanalysis	�#subanalysesCsubzsu�studying)
studystudies
studied
stuart�!structures�structure�%structurally!structuralh
struc�strongman�strongly2strongest�strongerEstrong�stringjstriking\strictly/'strictlenient/stricter|strict�striated�+stretchmediatedb!stretching�stretched#-stretchactivated�stretch�'stressrelated�+stressmimicking�'stressinduced�stresses
stressed�-stressassociatedfstressI%streptomycin.'strengthpower   kstrengthoriented
3strengthhypertrophy/-strengthhypertro�%strengthened
q!strengthen	�-strengthoriented
3strengthhypertrophy/superslow�surfaceasurfaces`symmetry]
suraeQ!supinationG#suppositionB!subdivided(
taper+supercompensate'theoretically�   v ����fG%	����kT8����aB����cE& ���|dD(
�
�
�
}
]
6
	�	�	�	d	F	����xV7����nR8����d?$����sN*����V;����yY;�����cG)����xV:               �R againRBadvmodis�Q !exercisingVBGpcompwith�P !parametersNNSpobjof�O interpretVBxcompdifficult�N necrosisNNconjdamage�M pervasiveJJamoddamage�L continuumNNdobjpropose�K proposeVBPROOTpropose�J hubalNNPconjhyldahl�I hyldahlNNPconjkinase�H secretedVBNconjescape�G escapeVBPrelclblood�F creatineNNcompoundkinase�E disturbedVBDaclresponse �D %inflammatoryJJamodresponse�C localJJamodresponse�B streamingNNpobjas�A zlineJJamodstreaming�@ #microscopicJJamodchanges�? repeatedVBNamodeffect�> #lengtheningVBGacldamage�= triggeredVBNrelcldamage�< globalJJamodmodels"�; #!dystrophiesNNSconjmyopathies�: abolishesVBZaclfact"�9 )putativeJJamodmechanosensors�8 knockoutNNnsubjabolishes�7 hamperedVBNconjare,�6 3mechanotransductionNNcompoundmechanism�5 freyNNPcompoundal�4 !propertiesNNSconjloading�3 %dependVBconjdemonstrated+�2 9httpsgtexportalorghomeNNadvmodtissues�1 lowestJJSpobjamong!�0 #!loadinducedJJamodactivation"�/ !%gadoliniumNNconjstreptomycin�. %streptomycinNNPpobjwith�- vivoJJpobjin�, ratsNNSpobjin!�+ #!nonspecificJJamodinhibition�* #mcbrideNNPconjspangenburg�) #spangenburgNNPconjpiezo�( piezoNNPapposgenes"�' %channelsNNSnsubjdemonstrated"�& #nucleiNNScompounddeformation"�% %neverthelessRBadvmodproteins"�$ !#shorteningVBGamodcontraction!�# !#myonuclearJJamoddeformation�" blockedVBNadvcloccur!�! !#yapinducedJJamodhypertrophy�  caveatsNNSattrare� cascadesNNSnsubjare+� /'deformingyapmtorcNNPconjfilaminbagyap"� 'filaminbagyapNNPpobjtogether� nucleusNNpobjto� cytosolNNpobjfrom� #translocateVBccompcauses � %intriguinglyRBadvmoddeform� !exposeVBconjsurrounded� desminNNcompoundfilaments!� %intermediateJJamodfilaments� filamentsNNSpobjby � tubulinNNcompoundfilaments� thickJJamodfilaments!� !!surroundedVBNROOTsurrounded(� +dephosphorylateNNPcompoundfilaminc� homodimerNNdobjdeform%� !'completelyRBadvmodcharacterized2� CfilaminbagmtorcyapautophagyJJcompoundcascade#� ##illustratedVBNROOTillustrated)� +!stimulussensingNNcompoundmechanisms � !!bagfocusedJJamodmechanisms$�
 )!aforementionedJJamodmechanisms�	 #contractingNNamodmuscle � %phosphatasesNNSconjkinases� mouseNNnmodmuscle%� 'highintensityNNcompoundexercise#� -phosphoproteomicJJamodstudies� %synaptopodinJJamodsynpo� #consequenceNNpobjas� normallyRBadvmodinhibit� awayRBadvmodtsc�  !sequestersNNSnsubjtsc� motifNNdobjbinds*�~ 7hypertrophyassociatedJJamodfunctions�} motifsNNScompoundmotifs#�| %separatedVBNrelcltryptophanes�{ %tryptophanesNNSpobjfor�z dimerNNnsubjactivates �y !influencesVBZrelclreceptor�x receptorNNpobjincluding �w androgenNNcompoundreceptor �v !!powerpointNNROOTpowerpoint)�u )!figuredownloadNNPcompoundpowerpoint'�t )downloadNNPcompoundfiguredownload�s damagedVBNamodproteins �r #degradationNNdobjregulates�q casaNNPdobjregulates�p !autophaghyNNPcompoundcasa!�o /chaperoneassistedJJamodcasa�n regulatesVBZrelclsynpo�m synpoNNpobjto�l encodesVBZrelclgene�k myonucleiNNpobjinto�j amotlNNSpobjas�i latsNNScompoundamotl�h domainsNNSpobjwith�g ppxyNNPcompounddomains �f #prolinerichNNPcompoundppxy!�e sequesterVBcompoundproteins�d wwNNPcompounddomain�c %intenseJJamodcontractions�b deformedJJacompbecomes�a bindsVBZrelclprotein�` #zdisklinkedVBNamodprotein�_ !referencesNNSpobjfor�^ schematicJJamodoverview�] attachedVBNrelclactin
	� �A	|	n	f	]	T	J	>
-�	1	&	�	
���
i�&S�|�4���6��xjRC6'| ����
�����	�	���b�	�v=i[PF8.%�H�1������& 
��

^
�Ts^A��QBp�:/�#��������
	�	�����|k	_QD:	�+�!�������q�������y�ocXB58.d%)������Vq�
te��}y	���rg[N�
�
��D�
�
�:/
�&

 ����
�
=��
�����M@�
IC�@��������xo	��f]
9≤	�−	�” �“ �	’s �’U‘\—8unfolds�'volumeequated
�vol
�≥
9≤	�−	�” �“ �	’s �’U‘\—8–%β�×	�°d
zones	�	zone	�	zinc	�zealand�	your�
young		york�
yield2yetV
yearsQ	yard�	xray
xwritingkwright�
would�
worth �working;	work�
words�	wolfwithout�'withinsubjectP/withinparticipant�within+withwwishes�	wishAwinningwingate�win9	will�	wilkW-widthorientation �
widthcwidely	wideD
whose�
whole who	�whilst�	whileE
which.whetherPwhereby	�whereas%
where�whenwhat'whaleywgt�!werkhausen�	were�welldevel
�	well �weights�weighted�weight�#weeksmonths�
weeks�weekly�	week �websitesC-webplotdigitizer�	weak�we:	ways
\way�
water	�waskwarrants
�warranted�warnekedwarmup'
waist�vsmvoluntary�volumes
d+volumedependent
evolume �vj�)visualizations�vigotsky�!viewpoints�viewpoint�viewing�viewed	view�viduals;victoria�	vicevic�viable
`via{	very<vertical�	verte
versus}versionsKversionp
versaveloped
�velocity�vastus�vastlyp	vast#vasculature�varyingb	vary �variousvariety�varies	�varied�!variationsvariation�variant	�variances
1variancecovariance�variance�variables-variable �#variability�vantages
T
vance
�
value?validity0valamatosfv�utilizing�utilized�utilising:usually%
usingfushaped
�useful�	used �use�usa�usAurementsxurementU� upright�%!velocities�!veragarciak%wellaccepted$workloads
veins�weakest�∼�weakly6yielded�
vitalkylänneY!usefulness(zdisks+αketoglutarate�
vitro�
vague�
women�vascular�virtuallyc
zlineA	vivo-!yapinduced!ww�#zdisklinked�	wwtrS'yesassociatedPyapO7vinculintalinintegrin!
zdiskvinculinvalues�µncell�µn�withstand�wonder�wk�workload�yr�	  �±� �� update�wherever�why�� understood�'wt−·day−�≈� undertaken2!weektoweek-T undulatingwinwood	vari�τ�vidual�� unweighted�wilson�volved�	valsmvancedfwholebodyPyousvention:� unravel(#welltrained$walker unanswered�xforce	user
vor)unin	!unilateral�uniformx%unidentified	�#undoubtedly
�%undetermined�undertake	*'understandingB!understandH%underpredict�%underpowered�underpin	�undergone)underdeveloped
�underde
�
under!ultimatelyNvshaped�U uptake�wildtype�%αβintegrin�� /ulum�+ultrastructural!ultrasound
u#zdisclinkedtJ junknowni#uncontestedBunclear�#uncertainty�uncertain�unbiased�#unavailable�
xweek�weaker�!volitionalW
valid%vieiraversial�� 	ure�validated�velopment�
wheth�
varie�
vasti�volves^workoutOworkoutsC!volumeload1ute-wasted
�'volumematched
�   untrained
�   w ����bA����^C$
����]?���wbD!����rW@
�
�
�
�
~
^
7
	�	�	�	�	j	1	����kK'����tK'���^7����xX7�����pV?%����v]4����gG'����rT8                                 �I decreasedVBDconjmeasured�H teschNNPauxmeasured�G biopsyNNcompoundstudy�F logicNNnsubjmeasured�E fatiguedJJamodmuscle�D #resynthesisNNconjdelivery�C deliveryNNdobjreduce�B !hydrolysisNNpobjof�A !biomarkersNNSnsubjchange�@ !glycolysisNNpobjthrough�? adp↔atpNNamodcreatine �> lohmannNNPcompoundreaction�= adpNNPnpadvmoddecline�< 'resynthesizesVBZpobjas$�; +oxidativeJJamodphosphorylation'�: '%resynthesizedVBNrelclcontractions�9 !hydrolyzedJJamodatp&�8 )%nonsteadystateJJamodcontractions�7 raisesVBZconjare�6 refutesVBZccompis�5 %powerliftersNNSpobjof�4 %eliteJJamodpowerlifters�3 exposedVBNccompsaying�2 sustainVBadvclrecruited�1 sayingVBGpcompof�0 proposalNNnsubjis�/ daNNPpobjbelow�. normoxiaNNPpobjvs�- hypoxiaNNpobjunder�, %intermittentJJamodhypoxia�+ phNNPnsubjmuscles�* pcrNNPpobjin�) +phosphocreatineJJamodpcr�( regimesNNSpobjin�' !invariablyRBadvmodoccurs�& %markedVBNaclcontractions�% vascularJJamodocclusion�$ !compensateVBccompsuggests#�# -occlusionrelatedJJamodstimuli�" occlusionNNpobjin�! atrophiedVBNccompoccluded�  thighsNNSdobjoccluded� occludedVBDadvclhave$� )intermittentlyRBadvmodoccluded� patientsNNSnsubjoccluded� surgeryNNcompoundpatients%� -braceimmobilizedVBNamodpatients!� 'postoperativeJJamodpatients� upstreamJJamodstimuli� middlemenNNSattrare!� +supplementationNNnsubjblunt&� #+antioxidantNNamodsupplementation� rosNNPapposspecies� speciesNNSnsubjpromote� oxygenNNcompoundspecies� reactiveJJamodspecies!� #bluntNNPcompoundhypertrophy� drugsNNSpobjas!� -antiinflammatoryJJamoddrugs� %nonsteroidalJJamoddrugs� !aidVBrelclproduction(� )!cyclooxygenaseNNcompoundproduction� myokinesNNSpobjincluding�
 !substancesNNSdobjproduce�	 !enterVBPconjassociated� immuneJJamodresponse6� MoxidemetalloproteinasehepatocyteJJcompoundfactor2� MnitricJJamodoxidemetalloproteinasehepatocyte� !stretchingNNconjexercise� quiescentJJamodcells� strongestJJSamodpathway� !coveredVBNadvclactivating� tooRBadvmodmany$�  -damageassociatedNNamodstimulus� #susceptibleJJamodmuscle&�~ ##eimdrelatedNNcompoundhypertrophy�} accordingVBGprepstep�| numbersNNSnsubjincrease�{ monthsNNSpobjfor�z !maintainedVBNadvcloccur �y %celldepletedVBNamodmuscles�x repairNNpobjin�w expandVBadvcllimit�v derivedVBDaclmyonuclei�u !presumablyRBadvmodderived"�t #respondedVBDrelclindividuals#�s '!proliferationNNconjactivation �r #proliferateVBadvclactivate �q #nondamagingVBGamodexercise�p adultNNcompoundmuscle�o addVBconjare#�n %nearlifelongNNPcompoundmuscle�m recipientNNcompoundmuscle �l +fiberassociatedJJamodcells�k 'transplantingVBGpcompas�j injuredVBDamodfibers�i #regeneratedJJamodfibers�h uninjuredJJamodfibers �g injectionNNcompoundresults#�f #cardiotoxinNNPcompoundresults�e %regenerationNNconjdamage�d cooccurNNPrelclstimuli!�c !virtuallyRBadvmodimpossible�b concludeVBxcompis�a therebyRBadvmodinhibit�` ampkNNdobjactivate!�_ #excessivelyRBadvmodactivate �^ %longdurationJJamodexercise�] %trainabilityNNdobjhave�\ anythingNNnsubjrunning�[ marathonNNnsubjrunning�Z sorenessNNconjlevels�Y plasmaNNPcompoundcreatine�X !cumulativeJJamodworkload�W !pretrainedVBNconjnaïve�V flannNNPcompoundal�U !attenuatesVBZconjgive�T !connectionNNdobjsuggest"�S )eimdassociatedJJamodstimulus   x ����lK5�����dL*����mC#���wV3����wZ? 
�
�
�
�
j
D
	�	�	�	�	�	Y	:		���aA!����_8����xO,����uYC&	����cD(����T9����b8����iA&             �A )experimentallyRBadvmodis�@ bigJJamodquestions%�? %#conclusivelyRBadvmodidentifying �> 'falsepositiveJJamodresults'�= )interpretationNNnsubjpasshampered�< feasibleJJacompsounds�; soundsVBZadvclbe$�: %spectrometryNNcompoundanalysis!�9 +exercisetrainedJJamodmuscle'�8 3coimmunoprecipitateVBadvclprovide�7 answersNNSdobjprovide�6 proteomicJJamodstudies�5 mediateVBPrelclproteins�4 interactVBaclknowledge �3 !physicallyRBadvmodinteract�2 identifyVBaclstrategy�1 challengeNNattris�0 problemsNNSnsubjbe*�/ 1hypertrophysensingNNcompoundfunction �. !modulatingVBGconjinducible�- inducibleJJccompmaking!�, %transgenesisNNdobjtargeting�+ !transgenicJJamodmodels�* 'sophisticatedJJamodmodels�) studyingVBGpcompfor�( !usefulnessNNdobjlimits�' limitsVBZrelclmyopathy�& myopathyNNnsubjpassneeded�% problemNNnsubjis�$ evaluateVBxcompinhibited(�# /pharmacologicallyRBadvmodinhibited�" knockedVBNxcompneeds�! !conclusiveJJacompare�  neverRBnegare� towardINprepproceed&� +synergisticallyRBadvmodinteracts� interactsVBZrelclspot"� 'damagerelatedJJconjmetabolic� spotNNattrbe� sweetJJamodspot� impactingVBGconjtrain � !interferesVBZrelclcapacity&� )forceproducingNNcompoundcapacity� impairsVBZadvclinhibit� hermeticJJamodcurve� certainlyRBadvmodfollow� realizedVBNadvclexists"� 'growthrelatedVBNamodbenefits� redundantJJconjsignaling$� '!manifestationNNnsubjcontribute(� +'traininginducedJJamodmanifestation� lendingVBGacleither� !transducedVBNconjsensed� answerVBxcompseek#� !filaminbagNNPcompoundproteins�
 spegNNconjobscurin�	 obscurinNNPapposkinases� localizedVBNamodkinases� robustlyRBadvmodalter� contractVBPadvclbecomes� zdisksNNPdobjsensing� genericJJamodadhesions$� 'partiallyRBadvmodcharacterized� potentJJconjlikely� exertVBrelclexercise�  happensVBZccompexplain&� %!energystressNNcompoundactivation�~ ampkαNNcompoundknockout�} inokiNNPcompoundal�| soonRBadvmodafter!�{ +aicarNNSdobjstressmimicking�z activatorNNcompoundaicar$�y +stressmimickingVBGaclmetabolic#�x !evolutionNNcompoundmechanisms,�w 3'phosphofructokinaseNNconjphosphorylase�v 'phosphorylaseNNpobjas�u inhibitedVBDconjinvolved�t fluxNNnsubjis�s rhebNNdobjbinds�r gapdhNNnsubjbinds�q !glycolyticJJamodenzyme�p hekNNcompoundcells�o tripleVBccompfeeding�n almostRBadvmodtriple�m feedingVBGadvclsensed�l mvpsNNPpobjby�k theorizedVBNconjcauses!�j #!halfmaximalJJamodactivation%�i #cellbasedVBNcompoundexperiments �h -lysophosphatidicJJamodacid�g placeboNNcompoundcontrol%�f +larginineNNPnmodsupplementation�e drinkingNNcompoundwater�d scavengerNNnsubjresulted!�c nitrogenNNcompoundscavenger�b citrateNNcompoundcycle�a +αketoglutarateNNattris'�` -!anabolismrelatedVBNamodmetabolite�_ modifierNNattrbe�^ receptorsNNSpobjthrough"�] 'membraneboundJJamodreceptors�\ initiateVBccompsuggests"�[ %gprdependentNNcompoundmanner�Z ohnoNNPcompoundal�Y )lactaterelatedJJamodgenes�X brooksNNPpobjof�W georgeNNPcompoundbrooks�V !laboratoryNNpobjfrom�U caffeineNNconjlactate�T %lowintensityNNnmodprogram�S vitroFWpobjin�R biomarkerNNattris!�Q 'stressrelatedVBNamodfactors�P serumNNpobjin�O reactionsNNSdobjcatalyze�N catalyzeVBPpcompgiven�M vagueJJamodconcept�L nonsteadyJJamodexercise�K womenNNSpobjin!�J #mmolkgNNcompoundpreexercise
� � �%����{ofVI:.#jJ
�
�
�
�
�
�
�
�
{
p
f
Z
O
A
.

	�	�	�	�	�	�	�	�	�	�	y	l	[	M	=	.	"		�����;�������x{[�h\N?."�
����������ufUI:)����������� �teR?3a'	����������}r0fXK:'	����������sd\QE8.$�	�����������~sbTE6&	������)�����~ndYL���������sTF7r�#eimdrelated~evidence%
every	�	everJ!eventuallyOeventualKevents�N
elite�
drugs�
enter�#eimdrelated~evidence%
every	�	everJ!eventuallyOeventualKevents�evenlyu	even0
evant	\evanston�ev<etitions�etitionethics�etc�	etal�etA+estimationbased�!estimation	Nestimates�estimated"estimate<#establishedc
estab�#essentially�essential	7essence�!especially%esmaeeldokhthescapeGesgery
�ersSerrors�
error�erload'erliftersH	eric�	ered�erector�ercise	z
erate

er	tequivocal�#equivalentsr!equivalent�equipment�
equip`equations�equating	�equatesequated
�equate0
equalqenzymes�enzymew%environments'environmental�#environment�environ�entiretyentire�
entedent�ensuring*ensure �	ensurenoughkenjoymentUenhancingZenhancesYenhanced\enhance �english^#engineering�engagingengage2energy)'energetically'endurancetype�endurance
endur�	endsendpoint�!endeavoredend<encourageEencodes�encoded*encode`
enced[	ence	�en$employingGemployed+
employ?#empiricallyYempirical
�empiri#emphasizingQ!emphasizesA!emphasized6emphasizeemphasis �emmeans�emergingaemerged�
email�em	�elusiveb#elucidating!elucidated	+elucidateOelszellite
i#eligibilityMeliciting�elicited�elicit�!elevationselevated
�elements�
elbowM!elasticity�elastic�
eitherS)eimdassociatedS	eimd�ei�egeficial)efforttrainingFefforts�effort �efficienti!efficiencyPefficacy#efficacious�effects�effectorsNeffector�'effectiveness �#effectively �effective �
effectmef�editingl
edema�ed1%ecologically$ecologi�'eccentriconly�'eccentrically�eccentricT
eccenKeasier�	ease�
earth�
early �earlier
earli�	each*dystrophy'9dystrophinglycoprotein 1dystrophinencoding##dystrophies;5dynamometerselectric�dynamic
dynamuduring)durationsEdurationR!duplicatesFdue;ducing�duchenne&
duced	%	duce)!dualenergy
wdu�dropsets���dropset�dropoff;	drop�drivers�driver
k
drive�	driv#
drink	l
draws�drawing�	draw�!downstream�downside�
downs�download�	downU
doubtdouble
doses	P%doseresponse	E	dose	Gdos'#dorsiflexedf	done�domised�dominate�domains�domain�
doings	does�doctoral[dodmd$dization�dividual�divided�diverseedivergentditions�ditional
/dition�disturbedE'distributions1%distribution
�#distributedv!distribute
�distribu�)distinguishing�#distinguish�dist#dynamometer�elevating|#dynamicallyNelevates)enableelevate
evoke�endocrine�elevationaelderlyIerk�!elongating�erroneous�!endomysial�enlarge}editedMevaluate$examiningexamined�examine+%energystress�#examination$	exam	a%exaggeratingexactly�
exactFex�evolved�evolution�-evidenceinformed	�   drinking�
� ���zm_SF�90F��� �����4�������xm_M9)
������CV��*������|seY&F6'��������, ����K�l���v
laMI;+^
��
�
�
��
�
�
�
�
�
u
d
S
>
4
&


l	�Z���b	��	�	�	�	�;vfT���v3S	�	�	�	�	}	s�	g	Z�	L~	>	2	$		
��:����<�L����xk]M>.!
��x����������a���y�kaTH;v3	%��)������c$o��q^J����;*	��������nitrogen�mo�����{qmlJmorgan�5!myopathies�!modulating.myopathy&
never 	mvps�nitrogen�modifier�#nonexistents�nonsteady�mmolkg�)nonsteadystate�normoxia�myofibers�%nonsteroidal�myokines�nitric�months{#nondamagingq%nearlifelongnnecrosisNmyf�'nonfunctional�myogenic�#nonspecific+%nevertheless%!myonuclear#
mousenormally
motif�motifs�myonuclei�)murfproteasome�mu�bolmyotraumac!nonhepaticQ%mobilizationN!mitigatingJ!modulationE%myogenically?nerves(/neurotransmitters&
named!modifiable�mitosis�myocyte�modulate�!nhterminal�modules�myocytes�
nodal�-mitogenactivated�#molecularly�!myogenesis�mrf�myogenin�	myod�	mrna�!myonucleus�mitotic�#mitotically�#miscounting�%northwestern�norrbrand/normal�nor=nonwarmup�nonvaried~!nonuniformo#nontraining�#nontargeted^nonstrict�)nonsignificant='nonresponders	�#nonreceptor,'nonperiodized�nonmuscle�
nonfiT!nonfailure#nonexercise
�#nonetheless	u	none�)noncontractile]noncancernologyA
noise�nogueirajno �nitionsRniques�nippard�	ning:nificant�nhouts�	next�new�!neutralarmQneuronsk'neuromuscular
Hneural�neu.network	�net2nested�
nents�nelles�
neity
neither<!negligibleCnegatives�!negativelynegative�negatedneg�
needsneeding�
neededN	need�necessity[necessary6#necessarily�
neces=nearly�	near�naïve	�nautilusnature
national �nation�narrower&narrative nancialUnamely	r	name�nal�navn�myotubes1myosin�!myofibrils�#myofibrillo�%myofibrillar�myofibril�
myofi�mwe	mvic�mvc	�mutations�mutation"	must
+musculoskeletal-#musculature4#muscularity�muscular �)musclespecificymuscles)musclebuilding
musclemus�multitudeBmultiset$!multiscaleYmultiple�)multinucleated�!multilevel�!multijointkmultihip,1multicompositionalZ
multi8	much%'mtorcmediated]
mtorc]	mtor8ms�mri�mpvzmpsPmpb	
moves.movements �movement]	moven
motorj
motion8mostlyh	most�mortality�!morphologyF'morphological0moreso$moreover�moreK
monte#'monophosphate�momentum�momentary �moment�molecules�molecularemodulateslmodifiesmodest+!moderatorsDmoderator�!moderation2)moderatevolume
�%moderateload	�moderate�
moder	�models1modelingsimulation�modeling�
model�	modexmodality�!modalities�mod
	mo�mm�	mize�%mixedeffects�
mixed	H'mitochondrial�mit
�missing�minutesWminute#minority!minimum
�!minimizing �minimize4   Zminimal�	ming�	mindmin,	milo�million}	milk	gmilder
,
nardone�minimal�	ming�	mindmoversZ!movementsaX#necessitateV#multiangledK#multiplanarJnonlinearInarrow0!motivation##multipleset
nerve�myoneural�#neutrophils�myoblasts�molecule��
mline�   �" ����gM/����oA#�����gH.�����jL0����uU>
�
�
�
�
�
l
O
4
	�	�	�	�	z	]	0	����qQ2 ����~gH"�����iS5����nQ1����}aG+����}P/����zX6����uZA"                          �A dnaNNPcompoundelements�@ sequenceNNpobjto�? mrfNNPconjmyogenin�> myogeninNNPpobjincluding�= myodNNPcompoundmyogenin�< myfNNPcompoundmyod�; !regulatoryJJamodfactors�: coexpressVBPROOTcoexpress �9 %proportionalJJamodincrease�8 #accompaniedVBNconjthought�7 !mrnaNNPcompoundproduction �6 !myonucleusNNnsubjregulates�5 proposesVBZrelclconcept�4 poolNNpobjas�3 !capabilityNNdobjretain�2 !mitoticJJamodcapability�1 retainVBPccomprequire�0 #mitoticallyRBadvmodcells*�/ ?nuclearcontenttofibermassJJamodratio�. extraJJamodnuclei�- donateVBPrelclone �, !precursorsNNSdobjproviding�+ createVBadvclfuse�* #fuseVBconjproliferate�) arousedVBNadvclimposed�( laminaNNpobjbetween�' basalJJamodlamina�& resideVBPrelclcells�% mediatedVBNxcompthought�$ carriedVBNrelclmass�# maintainVBconjavoid�" apoptosisNNdobjavoid�! #replacementNNdobjundergo�  undergoVBccompmeaning� #postmitoticJJamodtissue� !elongatingVBGpcompof� %arrangementsNNSpobjof!� %intricateJJamodarrangements� #miscountingNNpobjto� erroneousJJacompbe� yieldedVBDROOTyielded� avianJJdobjused� kelleyNNPpobjby� partlyRBadvmodbecause� putVBNROOTput� 'nonfunctionalJJamodas� !endomysialJJamodtissue� fibrousJJamodtissue� #perpetuatedVBNaclbelief� beliefNNdobjtraining � #concomitantJJamodincreases#� %augmentedVBNccomphypothesized� descendedVBDrelclthose� countNNdobjhad� inclineNNpobjon�
 treadmillNNnmodincline�	 climbedVBDadvclhad� morganNNPconjlynn� lynnNNPpobjin� castNNpobjin� serialJJamodincrease� diameterNNdobjaugments� augmentsVBZROOTaugments� myogenicJJamodevents� myofibersNNSpobjin �  'perturbationsNNSdobjcauses� subjectedVBNadvclaugments�~ !expandsVBZconjconsidered�} !enlargeVBPoprdconsidered*�| 3hypertrophyspecificNNcompoundroutine�{ achievingVBGpcompfor�z dictatesVBZROOTdictates�y attainedVBNadvcldictates!�x #!heighteningVBGaclexperience�w genderNNnsubjpassshown�v dominantJJamodfactor�u beginsVBZadvclis�t coupleNNpobjwithin�s #nonexistentJJacompis�r #populationsNNSpobjof�q %maximizationNNnsubjhas�p fullestJJSpobjto�o physiquesNNSdobjdevelop�n aspireVBPrelcllifters�m quantityNNpobjon#�l #competitorsNNSnsubjpassjudged�k vitalJJacompis�j footballNNcompoundrugby�i #correlationNNpobjgiven�h #extensivelyRBadvmodreview�g twofoldRBadvmodis!�f !#impressiveJJamodmuscularity�e lengthyJJamodperiods�d routinelyRBadvmodtrain�c fairlyRBadvmodshort�b pursuedVBNROOTpursued�a #resNNPapposapplication�` condNNPdobjtraining�_ !bjINprepschoenfeld�^ adviceNNpobjfor�] alexanderNNPconjchen�\ mattNNPcompoundalexander�[ chenNNPROOTchen�Z juNNPcompoundchen�Y ylänneNNPcompoundchen�X jariNNPcompoundchen�W thankVBPROOTthank�V finlandNNPpobjat�U #jyväskyläNNPpobjof�T octoberNNPpobjof�S placeNNdobjtook�R !tookVBDrelclphysiology�Q symposiumNNpobjduring+�P ++acknowledgmentsNNSROOTacknowledgments�O approvedVBDamodversion�N !revisedVBNamodmanuscript�M editedVBDROOTedited�L !draftedVBDamodmanuscript�K jjhNNPconjml�J mlNNPdobjbjs�I dlhNNPcompoundml�H preparedJJamodfigures�G !hwUHdetmanuscript�F declaredVBNROOTdeclared�E otherwiseRBadvmoddeclared#�D ##disclosuresNNSROOTdisclosures�C quoNNpobjon�B explainsVBZrelclthis   z ����nK%����^E&�����mO+����nQ1����f>
�
�
�
�
|
\
?
 	�	�	�	�	�	d	H	(		����z]B#�����iB#����wV="����pR1����lT4����gA"
����kS5����vX5                        �; !transportsVBZrelclchange �: )conformationalJJamodchange�9 cytoplasmJJapposcells�8 )disassociatingVBGpcompby#�7 )rapidlyRBadvmoddisassociating�6 weaklyRBadvmodbound �5 %biologicallyRBadvmodactive�4 unboundJJamodstate�3 globulinNNconjalbumin�2 steroidNNnmodglobulin�1 albuminNNpobjto�0 #boundVBNconjsynthesized�/ adrenalsNNSconjovaries�. ovariesNNSpobjfrom�- axisNNpobjvia,�, EhypothalamicpituitarygonadalJJamodaxis�+ testesNNSpobjof�* leydigNNPcompoundcells#�) ##synthesizedVBNROOTsynthesized�( nervesNNSdobjreleased�' !regenerateNNamodnerves �& /neurotransmittersNNSpobjof�% %considerableJJamodeffect%�$ 1cholesterolderivedJJamodhormone!�# !channelNNcompoundexpression�" ltypeNNPcompoundchannel�! ratiosNNSpobjto �  %helpingVBGconjfacilitating � %donationNNdobjfacilitating� fusionNNdobjenhance � 'igfieaNNPconjproliferation� locallyRBadvmodexpressed� promotesVBZadvclexerts� paracrineNNconjautocrine� autocrineJJnmodmanner� #upregulatedVBNccompis � !thereafterRBadvmodelevated� splicesNNSnsubjremain� startsNNSnmodmuscle� kickNNnmodstarts� splicedVBNccompcauses� igfsNNcompoundmode� mgfNNPoprdcalled� mechanoNNPcompoundfactor� !familiarlyRBadvmodcalled� igfecNNconjigfeb� igfebNNPconjigfea� igfeaNNPapposforms$� !isoformsNNSnsubjpassidentified�
 igfbpNNpobjto�	 igfbpsRBadvmodproteins$� #%circulatingVBGamodavailability� liverNNpobjthan� schwannNNcompoundcells#� %similaritiesNNSnsubjpassfound� namedVBNadvclis� peptideNNcompoundhormone!� %structurallyRBadvmodloading� prematureJJconjcontext�  dismissalNNnsubjis� overtJJamoddismissal �~ !questionedVBNccompprovides�} ghINadvclshown�| mediatingVBGpcompin�{ addressedVBNconjbelieved�z regimenNNpobjof�y !modifiableJJamodaspect�x !suppressedVBNpcompgiven�w mitosisNNdobjinduce"�v ##proteolysisNNdobjattenuating�u #attenuatingVBGpcompon�t possessVBxcompshown�s insulinNNdobjpromote�r !inhibitoryJJamodfactor�q leukemiaNNcompoundfactor�p !fibroblastJJamodgrowth�o #interleukinNNPnmodil�n hepatoNNPcompoundfactor$�m %%facilitatingVBGaclinteractions�l !likelihoodNNdobjincrease�k servingVBGaclresponse�j integralJJamodrole�i cytokinesNNSconjhormones�h hormonesNNSROOThormones%�g #cndependentNNPcompoundsignaling"�f 'gataNNPcompoundtranscription�e myocyteJJamodfactor�d mediatesVBZconjbelieved�c actsVBZxcompsignaling$�b #phosphataseNNnsubjpassbelieved"�a ##caregulatedJJamodphosphatase�` !cnNNPnpadvmodimplicated�_ #calcineurinNNPpobjof!�^ !!implicatedVBNROOTimplicated�] #cadependentJJamodpathways$�\ -calciumdependentJJamodpathways�[ modulateVBPrelclfactors�Z rapidJJamodrise�Y jnkNNPapposmapk!�X !nhterminalNNPcompoundkinase�W cjunNNPcompoundkinase�V erkNNPcompoundmapk#�U +signalregulatedVBNamodkinases�T modulesNNSconjgrowth�S myocytesNNSpobjin�R redoxJJamodstatus$�Q 'proteinkinaseJJcompoundpathway�P catabolicJJamodsignals�O nodalJJamodpoint�N !aktNNPnsubjpassconsidered�M aktmtorJJcompoundpathway�L calciumcaNNconjmapk�K mapkNNPnmodpathways(�J -mitogenactivatedNNPcompoundprotein�I %aktmammalianJJamodtarget#�H #!molecularlyRBadvmodtransduced �G 1mechanostimulationNNpobjof#�F ##facilitatedVBNROOTfacilitated�E !myogenesisNNpobjin�D rolesNNSdobjplaying�C playingVBGpcompwith�B promoterNNpobjin   { ���pL0����qL1����pQ)�����tW:����mS7
�
�
�
�
�
m
O
0
	�	�	�	�	a	F	*	���|_=�����xV7����y[>�����kS0����wK)	���|`>�����nO/�����cH,                 �6 tearingNNpobjbecause�5 #homeostasisNNpobjof�4 !disruptionNNpobjto�3 ttubulesNNSdobjdeforms�2 deformsVBZrelclshearing�1 shearingNNdobjcauses�0 regionsNNSpobjat�/ !weakestJJSamodsarcomeres�. !supportiveJJamodtissue�- tearsNNSpobjin�, )macromoleculesNNSpobjto�+ exhibitedVBDccompfound�* rabbitsNNSpobjin�) pradoNNPcompoundal!�( !slowtwitchNNPcompoundfibers�' pkcVBconjcamkiv�& camkivNNconjcamkii�% camkiiNNPdobjencode&�$ !#calmodulinNNPcompoundcalcineurin�# firingNNcompoundfrequency�" muNNPcompoundfiring�! couplingNNpobjof"�  !excitationNNcompoundcoupling$� !amplitudeNNnsubjpassdetermined%� /extramyofibrillarJJamodelements� developsVBZROOTdevelops� regulatedVBNccompsuggests)� /!mechanochemicallyRBadvmodtransduced� disturbsVBZccompbelieved� #translationNNpobjof � unloadingVBGadvclincreases� !pronouncedVBNamodeffect� !generationNNpobjby� #responsibleJJacompare � !!initiationNNROOTinitiation� agentsNNSpobjof� endocrineJJamodagents� ischemicJJamodexercise� hyperemiaNNpobjfrom� myoblastsNNSpobjin� cardiacJJamodmuscle� smoothJJamodmuscle%� )#hypoxicinducedJJamodhypertrophy� cytokineNNcompoundil�
 hypoxicJJamodtraining�	 clearanceNNcompoundrate� theoriesNNSattrare&� +mediumintensityNNcompoundflexion� maimumNNPcompoundrm� !∼NNPcompoundrepetition� subgroupsNNSpobjinto� !protectiveJJamodeffect� !conferringVBGaclocclusion� kubotaNNPcompoundal�  bedNNpobjto� confinedVBNaclgroup�~ takaradaNNPcompoundal�} storesNNSdobjpossess�| reflectVBrelclpotential�{ attractsVBZpcompgiven�z storageNNcompoundcapacity�y regimensNNSnsubjhave�x !anaerobicJJamodglycolysis�w %influxNNdobjfacilitating%�v 3oxidativeglycolyticJJamodfibers#�u aquaporinNNPcompoundaquaporin�t osmoticJJamodchanges�s #contributorNNpobjas�r actingVBGpcompwith�q reliesVBZrelclexercise�p maximizedVBNconjshown*�o AalphamethylaminoisobutyricJJamodacid$�n 1integrinassociatedJJamodsensor �m transportNNcompoundsystems�l hydratedJJamodcell�k )ultrastructureNNpobjof�j 'reinforcementNNpobjto�i integrityNNpobjto�h threatNNpobjas"�g pressureNNnsubjpassperceived�f linkingVBGaclbasis�e simulateVBxcompknown�d hydrationNNnsubjserves�c myotraumaNNPpobjwith�b #conjunctionNNpobjin�a elevationNNpobjof�` spikesNNSdobjreplicate�_ %administeredVBNadvclfind�^ failedVBNrelclstudies�] !postulatedVBNconjspike�\ spikeVBPROOTspike�[ boneNNnmodmodeling�Z exertingVBGpcompto�Y halflivesNNSpobjwith�X kdaNNcompoundisoform�W sleepNNpobjduring�V !secretionsNNSpobjwith�U pulsatileJJamodfashion�T glandNNpobjby�S pituitaryJJamodgland(�R 1autocrineparacrineNNcompoundmanner%�Q !!nonhepaticNNPcompoundexpression�P #upregulatesVBZconjacts�O 'incorporationNNconjuptake�N %mobilizationNNpobjtoward�M agentNNpobjas$�L )repartitioningVBGcompoundagent�K #polypeptideNNamodhormone�J !mitigatingVBGaclelderly�I elderlyJJconjwomen"�H secretionNNcompoundahtiainen!�G !fasttwitchNNcompoundmuscles�F fibertypeNNnmodmanner�E !modulationNNpobjin�D !upregulateVBxcompshown�C !compromiseVBxcompshown!�B !seriouslyRBadvmodcompromise�A #suppressionNNpobjof!�@ #committedVBNamodsuppression#�? %myogenicallyRBadvmodcommitted#�> !%inhibitingVBGamodtestosterone�= magnifiedVBNROOTmagnified�< #chromosomalNNPcompounddna
   l* ����?���wl_M:1#	��*���~vh�����yeO9*
���������xbbvM5
�����pe�[�OJ<,
�
�
�
�
�
�
�
�
�
�
w
[
<
.�

	�	�	�	�	�	�	�	�	�	t	g	U	L	=	2	)		���                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          +shepstone et al�!schieppati�'three seconds�)the fifth week�-the first minute�!veragarcia�7those with plus years�!slowtwitch�#three grams�zdisks�
zdiskM$Kzdisclinked phospholipase d pld\	years
yap  jyapQ
xweekwwtr  yapRwolf%within  daystwinwood1%wilson et al"wilk
wheth �%whaley et alK1wgt cmjsprint timeL-werkhausen et alN!weektoweek;	weeksQ
weekly4	week �)vinculin talinL)vic  australia �valamatos??us national dairy council?;upperbody muscle groupsnupperbody[#up to  daysaunderde �un �u  °  °Etwottnctsc mtorOtscf5tra ditional lowload �1torgluteus maximusIton al>tidybayes:#ticipant  lMticipant
C#three weeks7	threea
thirdI?thigh crosssectional area7the successive  weeks1the rd and th week �1the preex haustion-the past century �7the mediation ef fect �*Ythe international prospective register,1the initial  weeks
+the full  weeks:-the final  weeks5the early first week �%the ac crual �
teschw#terventions\ten �taz yapXtazS!tal muscle �sys temicsys tem �sys �/subse quent elbow �su perset -strengthhypertro �/stasinaki et al  j)sta tistically%sportsdiscus&?sport victoria university~sponse �!sociations �J�social sciences solent university southampton uk institute for health}X�3social sciences solent university southampton uk city university of new york lehman%six weekslsixesimi lar �*Wsig nificant betweengroup differences5+shortterm hours �3shortermusclelength'several weeks �%several daysb7sessionsexercisessets;seriou saUseriS'senna et al   �%selectorizedsecondlyb
second6sec$Kscien tific research mechanical �%schoenfeld  *!schoenfeldsatog!sarcoplasm �#sar coplasm �#rt programs �   { ����oR4����xU7����{V7����iG'����gK*
�
�
�
�
w
_
5
	�	�	�	�	o	P	6	�����kF�����iJ)���yU;����yZ>�����jE#����{`I)����jB# ����lO5                     �1 deltoidJJcompoundactivity�0 narrowJJamodbench�/ !clavicularJJamodhead�. sternalJJamodhead�- !pectoralisNNPpobjto�, depressesVBZconjabducts�+ abductsVBZccompelevates�* scapulaNNdobjelevates �) !elevatesVBZadvclsubdivided�( !subdividedVBNccomphave%�' !trapeziusNNPnsubjpasssubdivided�& leverageNNdobjprovide�% %compartmentsNNSpobjwithin�$ %wellacceptedJJamodtenet�# !motivationNNdobjbalanced�" !arisingVBGacldecrements�! !decrementsNNSpobjwith�  balancedVBNROOTbalanced� adjustVBconjbe� cognizantJJacompbe� abilitiesNNSnsubjare� !highvolumeNNPconjis� traumaNNattris� !repetitiveJJamodtrauma� statesNNSpobjof� cortisolNNcompoundlevels� #luteinizingVBGamodhormone"� #chronicallyRBadvmoddecreased#� #%overtrainedJJamodovertraining� cessationNNconjtaper� taperNNpobjby� plusJJintjwith"� +supercompensateVBccompcauses� reboundNNcompoundeffect� #culminatingVBGaclcycle� enableVBccompallows� affordedVBNaclrecovery� splitJJamodroutine� mhNNcompoundroutine�
 smiliosNNPcompoundal�	 !indicatingVBGaclset� fourthJJamodset� !completionNNpobjafter� untilINprepincrease� schwabNNPpobjthan!� lowvolumeNNcompoundroutines� elevateVBxcompproven� workloadsNNSpobjof%� #multiplesetNNPcompoundprotocols&�  %highervolumeNNPcompoundprotocols$� 'consecutivelyRBadvmodperformed�~ #impracticalJJccompmaking�} !borneVBNconjpostulated�| !intriguingJJacompis�{ !containingVBGaclmuscles�z girthNNdobjmaximize�y profilesNNSdobjexhibit�x exhibitVBPpcompgiven�w 'applicabilityNNdobjhave&�v )#fatigueabilityNNPconjmicrotrauma"�u 'theoreticallyRBadvmodenhance�t schemeNNpobjwith�s osmolytesNNSnsubjdrawing�r !byproductsNNSpobjof�q gradientNNdobjcauses�p spacesNNSpobjinto�o %interstitialJJamodspaces�n #capillariesNNSpobjof�m seepVBccompcauses�l deliverVBxcompcontinue�k arteriesNNSnsubjcontinue�j !compressedJJacompare�i veinsNNSdobjtraining�h maximizesVBZccompcauses �g !!remodelingNNdobjfacilitate%�f )#tensioninducedJJamodhypertrophy'�e -glucosephosphateNNcompoundbuildup�d glucoseNNpobjin�c declinesNNSdobjshow*�b /bodybuildingstyleNNPcompoundexercise�a relyVBPROOTrely�` optimizesVBZaclbelief�_ evokeVBPcsubjbeen�^ repsNNSnsubjevoke�] !recruitVBxcompinadequate�\ !inadequateJJacompis�[ !bringVBadvclconsidered!�Z %artificiallyRBadvmodinduced�Y provenVBNROOTproven�X taxVBconjinvolve �W !classifiedVBNconjexpressed"�V #customarilyRBadvmodexpressed�U nerveNNcompoundactivity�T #sympatheticJJamodactivity�S #acidicJJamodenvironment"�R )growthorientedVBNamodfactors!�Q #!freeradicalJJamodproduction�P milieuNNcompoundcell�O #hypertophicJJamodresponse#�N '!stressinducedJJamodmechanisms�M ischemiaNNconjcreatine"�L phosphateNNPcompoundcreatine"�K inorganicNNPcompoundcreatine�J hydrogenNNcompoundion�I buildupNNpobjin�H manifestsVBZROOTmanifests�G heightenVBxcompintended�F secondaryJJconjprimary�E impingingVBGaclnerves �D 'givesVBZrelclconcentration�C junctionNNpobjunder�B myoneuralJJamodjunction�A debrisNNdobjremove�@ removeVBconjreleased#�? ##lymphocytesNNSconjmacrophages�> #macrophagesNNSdobjattract�= attractVBPrelclfibers�< #microtraumaNNPpobjof�; migrateVBPadvcllikened�: #neutrophilsNNSpobjby�9 infectionNNpobjto�8 likenedVBNROOTlikened�7 openingNNdobjinduces   pV ����xW/����hJ0����y]<����kE#���`H#
�
�
�
�
j
K
1
	�	�	�	�	w	U	5	 	����oT0����mQ5�����fE$����bD)����}Y;#����bF!���{V                                                                                                                                                                                                                                                                                                                                                                              "�! !!culminatesVBZadvclperiodized#�  ##microcyclesNNSdobjalternating%� %%applicationsNNSROOTapplications� aidedVBNrelcladvantage� resistiveJJamodforce� #dynamometerNNpobjon"� 'forcevelocityNNcompoundcurve� rad·s−JJdobjslow"� !#rads·s−NNSnmodrepetitions� shepstoneNNPcompoundal� gastrocNNPpobjof� soleusNNPcompoundmuscle � 'derecruitmentNNPdobjshowed� !schieppatiNNPconjnardone� nardoneNNpobjby� reversalNNpobjbecause!� superslowNNcompoundtraining� !velocitiesNNSpobjat� !augmentingVBGadvclenhance� demandNNnsubjpassshown� !heightenedJJamoddemand� speedsNNSpobjat� cadenceNNpobjat�
 drawnVBNconjimpact�	 performsVBZrelclspeed� !bluntingNNconjreductions� izquierdoNNPcompoundal� %burnoutNNconjovertraining � 'psychologicalJJamodburnout� linnamoNNPcompoundal� heightensVBZconjenhance� !continuingVBGaclresponse� %potentiatingVBGaclstress�  meritsNNSnsubjare � )concentricallyRBadvmodlift �~ %postadaptiveJJamodresponse�} fifthJJamodweek�| elevatingVBGpcompon�{ bureshNNcompoundal�z bodysNNcompoundanabolic�y shuttleVBconjbuffer�x bufferVBaclcapacity�w capillaryJJamoddensity�v lifterNNnsubjsustain �u recoveredVBNccompindicates"�t %!satisfactoryJJamodcompromise!�s #compromisedVBNconjmaximized�r sallesFWcompoundal!�q +counterbalancedVBNccomptend�p regainVBadvclallow�o !developingVBGpcompin�n !upsNNSdobjperforming�m obliquesNNSconjabdominis�l abdominisNNPSpobjof�k !veragarciaNNPcompoundal�j abdominusNNpobjin�i ballNNpobjon�h stabilityNNcompoundball�g crunchesNNSnsubjperformed�f !sternlichtNNPcompoundal�e exceptionNNnsubjinvolves!�d %diminishVBadvcldemonstrated�c behmNNPconjanderson�b andersonNNPnpadvmodcarry�a surfaceNNpobjon�` surfacesNNSpobjof�_ unstableJJamodsurfaces!�^ %architectureNNnsubjsuggests�] symmetryNNdobjimproving�\ imbalanceNNdobjcreating�[ !precedenceNNdobjtake�Z moversNNSnsubjtake�Y #inefficientJJacompis"�X !movementsaNNcompoundstrategy�W dozensNNSpobjof�V #necessitateVBccomptend�U coverageNNpobjof!�T 'posturalJJamodstabilization�S rhomboidsNNPpobjincluding$�R !abdominalsNNScompoundrhomboids�Q suraeNNPconjabductors�P abductorsNNSpobjincluding#�O adductorsNNScompoundabductors!�N #dynamicallyRBadvmodrecruits �M 'stabilizationNNdobjrequire�L producingVBGaclmovements �K #multiangledVBNamodapproach�J #multiplanarJJamodapproach!�I %nonlinearJJamodcombinations�H centrallyRBadvmodlocated�G !supinationNNpobjfor�F %partitioningNNpobjof�E insertionVBadvclspan�D #spanVBPaclsupposition�C alwaysRBadvmodspan�B #suppositionNNdobjrefuting�A refutingVBGxcompterminate�@ branchesNNSpobjby�? #compartmentNNpobjwith�> %inscriptionsNNSconjbands�= bandsNNSpobjby �< gracilisNNPcompoundfemoris"�; sartoriusNNPcompoundgracilis �: portionsNNSnsubjpasscalled!�9 -branchsuggestingVBGaclnerve%�8 1componentsdistinctJJamodregions�7 sometimesRBadvmoddivided�6 inactiveJJacompis�5 adjacentJJamodfiber�4 scatteredVBNconjimpact�3 fastJJconjslow�2 #inclinationNNpobjof
 ��������[9�vfUC0%�������.�h��P-��sjdSA5'�����������sdSC4)1�����������<�{Jrh[RF8(;������A�����������|o\L?0xh��		
O����#�	���������ugZK>1#s]�����X��!�����~sg\RD9+OfE���������desm     delivery�' dadiscusses|� )damagerelated-damageassociated�derivedvdmd$ur!desm     delivery�' dadiscusses|� damagerelateddemand�
drawn�derivedvdmd$ur!discussion'!discussing#depend3� 'derecruitment�disease�desmin+dephosphorylate
dimer�\ 8damaged�#degradation�\ deformed�deform�#discussionsF�depends�)differentiated�display
�y definite�described�� codiseases�
dgkξ})diacylglycerol{dioydigitized�difficultddiffering�#differently[+differentiationZdifferent �#differences�!differencediffered�differldif
�dietary&	diet�didI%dichotomized�diator�devoteddevote�devices�device5!deviations�deviation�%developments#development �developed	develop
devel�deuterium
�!deuterated	�#detrimental;detriment�detrimenB#determining�!determined
�determineOdetectedSdetect�detailsUdetailedndetail�despite=desired�desirabledesignsdesigned"design�%descriptionsp#description�describes �describe �descent�!descending�
depth�depending �dependent�!dependence�#departments�!department�'deoxygenationdensity4denotes\
dence	�'demonstrating�%demonstrated�#demonstrate	�!demonstrat�demands]demanding6
delta�deloads9delineateMdeline#deleterious�delayed�� �degreesAdegree�� �deformationinitiatedp#deformation�#definitions`!definition �!definitely�defined�define	defiQdefault:deemedr� +diseasespecific	 decreased�Ado-disproportionate�#displeasure
Odisplays�displayed�%displacement}wdecrement�!decreasing decreases�decreaseqdecline;declare�%declarations�decision=decipherdeciding-decades{debates�debatedOdebate�dearthOdeadlifts0deadlift�dejdaysweek�#daystoweeks	�	days
�dayh
daugh�datingS	datedatasheet�)datagenerating�databasesgdatabase�	dataL9damageinjuryassociatededamage	�dality6dalities�   @dairyO
daily<d�cycling�cycles
�
cycle!developingodiminishddozensWdeltoid1depresses,� decrements!deliver�� declines�debris�!disruption�| deforms�develops�disturbs�)disassociating8donationdismissal dna�donate�descended�diameter�dictateszdominantvdraftedLdlhI   declaredF#disclosuresDdropoff;	drop�drivers�driver
k
drive�	driv#drinking�
drink	l
draws�drawing�	draw�!downstream�downside�
downs�download�	downU
doubtdouble
doses	P%doseresponse	E	dose	Gdos'#dorsiflexedf	done�domised�dominate�domains�domain�
doings	does�doctoral[dization�dividual�divided�diverseedivergentditions�ditional
/dition�disturbedE'distributions1%distribution
�#distributedv!distribute
�distribu�)distinguishing�#distinguish�distinctodistal dispute�discussed
!discussd!discrepanto'discrepanciesdiscrep	�!discovered
!discomfort
N#discerniblebdis�directs�directly �!directions�'directionally9direct�