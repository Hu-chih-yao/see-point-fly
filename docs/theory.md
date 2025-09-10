3. Prompting with Iterative Visual Optimization
The type of tasks this work considers have to be solved by producing a value  ∈ A from a set A
given a task description in natural language  ∈ L and an image observation  ∈ ℝ××3
. This set
A can, for example, include continuous coordinates, 3D spatial locations, robot control actions, or
trajectories. When A is the set of robot actions, this amounts to finding a policy (·|, ) that emits
an action  ∈ A. The majority of our experiments focus on finding a control policy for robot actions.
Therefore, in the following, we present our method of PIVOT with this use-case in mind. However,
PIVOT is a general algorithm to generate (continuous) outputs from a VLM.
3
PIVOT: Iterative Visual Prompting Elicits Actionable Knowledge for VLMs
3.1. Grounding VLMs to Robot Actions through Image Annotations
We propose framing the problem of creating a policy  as a Visual Question Answering (VQA) problem.
The class of VLMs we use in this work take as input an image  and a textual prefix  from which
they generate a distribution VLM(·|, ) of textual completions. Utilizing this interface to derive a
policy raises the challenge of how an action from a (continuous) space A can be represented as a
textual completion.
The core idea of this work is to lift low-level actions into the visual language of a VLM, i.e., a
combination of images and text, such that it is closer to the training distribution of general visionlanguage tasks. To achieve this, we propose the visual prompt mapping

ˆ, 1:

= Ω(, 1:) (1)
that transforms an image observation  and set of candidate actions 1:,  ∈ A into an annotated
image ˆ and their corresponding textual labels 1: where  refers to the annotation representing
in the image space. For example, as visualized in Fig. 1, utilizing the camera matrices, we can project
a 3D location into the image space, and draw a visual marker at this projected location. Labeling
this marker with a textual reference, e.g., a number, consequently enables the VLM to not only be
queried in its natural input space, namely images and text, but also to refer to spatial concepts in its
natural output space by producing text that references the marker labels. In Section 4.4 we investigate
different choices of the mapping (1) and ablate its influence on performance.
3.2. Prompting with Iterative Visual Optimization
Representing (continuous) robot actions and spatial concepts in image space with their associated
textual labels allows us to query the VLM VLM to judge if an action would be promising in solving
the task. Therefore, we can view obtaining a policy  as solving the optimization problem
max
∈A,
VLM


 ˆ, 
s.t.
ˆ, 
= Ω(, ). (2)
Intuitively, we aim to find an action  for which the VLM would choose the corresponding label
after applying the mapping Ω. In order to solve (2), we propose an iterative algorithm, which we refer
to as Prompting with Iterative Visual Optimization. In each iteration  the algorithm first samples
a set of candidate actions
()
1:
from a distribution A() (Figure 2 (a)). These candidate actions are
then mapped onto the image  producing the annotated image ˆ
() and the associated action labels

()
1:
(Figure 2 (b)). We then query the VLM on a multiple choice-style question on the labels
()
1:
to choose which of the candidate actions are most promising (Figure 2 (c)). This leads to set of
best actions to which we fit a new distribution A(+1) (Figure 2 (d)). The process is repeated until
convergence or a maximum number of steps  is reached. Algorithm 1 and Figure 2 visualize this
process.
3.3. Robust PIVOT with Parallel Calls
VLMs can make mistakes, causing PIVOT to select actions in sub-optimal regions. To improve the
robustness of PIVOT, we use a parallel call strategy, where we first execute  parallel PIVOT instances
and obtain  candidate actions. We then aggregate the selected candidates to identify the final
action output. To aggregate the candidate actions from different PIVOT instances, we compare two
approaches: 1) we fit a new action distribution from the  action candidates and return the fitted
action distribution, 2) we query the VLM again to select the single best action from the  actions. We
PIVOT: Iterative Visual Prompting Elicits Actionable Knowledge for VLMs
find that by adopting parallel calls we can effectively improve the robustness