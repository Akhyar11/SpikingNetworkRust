TMLR Ethics Guidelines

Adapted from the NeurIPS Ethics Guidelines. For additional guidelines on ethical publishing, see our editorial policies.
Introduction

As ML research and applications have increasing real-world impact, the likelihood of meaningful social benefit increases, as does the attendant risk of harm. Indeed, problems with data privacy, algorithmic bias, automation risk, and potential malicious uses of AI have been well-documented [1].

In the light of these findings, machine learning (ML) researchers can no longer ‘simply assume that... research will have a net positive impact on the world’ [2]. The research community should consider not only the potential benefits but also the potential negative impacts of ML research, and adopt measures that enable positive trajectories to unfold while mitigating risk of harm. We expect authors to discuss such ethical and societal consequences of their work in their papers, while avoiding excessive speculation.

This document should be used by both authors and reviewers to establish common ground about the ethics guidelines. The primary goal of the Broader Impact Concerns assessment by reviewers is to provide critical feedback for the authors to incorporate to improve their paper. In rare situations, however, TMLR reserves the right to reject submissions that have violated key ethical principles.

There are two aspects of ethics to consider: potential negative societal impacts (Section 2) and general ethical conduct (Section 3).
Potential for Negative Societal Impact

Submissions to TMLR are expected, when applicable, to include a discussion about potential negative societal impacts of the proposed research artifact or application. Whenever these are identified, submissions should also include a discussion about how these risks can be mitigated.

Grappling with ethics is a difficult problem for the field, and thinking about ethics is still relatively new to many authors. Given its controversial nature, we choose to place a strong emphasis on transparency. In certain cases, it will not be possible to draw a bright line between ethical and unethical research. A paper should therefore discuss any potential issues, welcoming a broader discussion that engages the whole community.

A common difficulty with assessing ethical impact is its indirectness: most papers focus on general-purpose methodologies (e.g., optimization algorithms), whereas ethical concerns are more apparent when considering deployed applications (e.g., surveillance systems). Also, real-world impact (both positive and negative) often emerges from the cumulative progress of many papers, so it is difficult to attribute the impact to an individual paper.

The ethics consequences of a paper can stem from either the methodology or the application. On the methodology side, for example, a new adversarial attack might give unbalanced power to malicious entities; in this case, defenses and other mitigation strategies would be expected, as is standard in computer security. On the application side, in some cases, the choice of application is incidental to the core contribution of the paper, and a potentially harmful application should be swapped out (as an extreme example, replacing ethnicity classification with bird classification), but the potential mis-uses should be still noted. In other cases, the core contribution might be inseparable from a questionable application (e.g., reconstructing a face given speech). In such cases, one should critically examine whether the scientific (and ethical) merits really outweigh the potential ethical harms.

A non-exhaustive list of potential negative societal impacts is included below. Consider whether the proposed methods and applications can:

    Directly facilitate injury to living beings. For example: could it be integrated into weapons or weapons systems?
    Raise safety or security concerns. For example: is there a risk that applications could cause serious accidents or open security vulnerabilities when deployed in real-world environments?
    Raise human rights concerns. For example: could the technology be used to discriminate, exclude, or otherwise negatively impact people, including impacts on the provision of vital services, such as healthcare and education, or limit access to opportunities like employment? Please consult the Toronto Declaration for further details.
    Have a detrimental effect on people’s livelihood or economic security. For example: have a detrimental effect on people’s autonomy, dignity, or privacy at work, or threaten their economic security (e.g., via automation or disrupting an industry)? Could it be used to increase worker surveillance, or impose conditions that present a risk to the health and safety of employees?
    Develop or extend harmful forms of surveillance. For example: could it be used to collect or analyze bulk surveillance data to predict immigration status or other protected categories, or be used in any kind of criminal profiling?
    Severely damage the environment. For example: would the application incentivize significant environmental harms such as deforestation, fossil fuel extraction, or pollution?
    Deceive people in ways that cause harm. For example: could the approach be used to facilitate deceptive interactions that would cause harms such as theft, fraud, or harassment? Could it be used to impersonate public figures to influence political processes, or as a tool of hate speech or abuse?

General Ethical Conduct

Submissions must adhere to ethical standards for responsible research practice and due diligence in the conduct.

If the research uses human-derived data, consider whether that data might:

    Contain any personally identifiable information or sensitive personally identifiable information. For instance, does the dataset use features or label information about individual names? Did people provide their consent on the collection of such data? Could the use of the data be degrading or embarrassing for some people?
    Contain information that could be deduced about individuals that they have not consented to share. For instance, a dataset for recommender systems could inadvertently disclose user information such as their name, depending on the features provided.
    Encode, contain, or potentially exacerbate bias against people of a certain gender, race, sexuality, or who have other protected characteristics. For instance, does the dataset represent the diversity of the community where the approach is intended to be deployed?
    Contain human subject experimentation and whether it has been reviewed and approved by a relevant oversight board. For instance, studies predicting characteristics (e.g., health status) from human data (e.g., contacts with people infected by COVID-19) are expected to have their studies reviewed by an ethical board.
    Have been discredited by the creators. For instance, the DukeMTMC-ReID dataset has been taken down and it should not be used in TMLR submissions.

In general, there are other issues related to data that are worthy of consideration and review. These include:

    Consent to use or share the data. Explain whether you have asked the data owner’s permission to use or share data and what the outcome was. Even if you did not receive consent, explain why this might be appropriate from an ethical standpoint. For instance, if the data was collected from a public forum, were its users asked consent to use the data they produced, and if not, why?
    Domain specific considerations when working with high-risk groups. For example, if the research involves work with minors or vulnerable adults, have the relevant safeguards been put in place?
    Filtering of offensive content. For instance, when collecting a dataset, how are the authors filtering offensive content such as racist language or violent imagery?
    Compliance with GDPR and other data-related regulations. For instance, if the authors collect human-derived data, what is the mechanism to guarantee individuals’ right to be forgotten (removed from the dataset)?

This list is not intended to be exhaustive — it is included here as a prompt for author and reviewer reflection.
Final Remarks

In summary, we expect TMLR submissions to include discussion about potential harms, malicious use, and other potential ethical concerns arising from the use of the proposed approach or application. We also expect authors to include a discussion about methods to mitigate such risks. Moreover, authors should adhere to best practices in their handling of data. Whenever there are risks associated with the proposed methods, methodology, application or data collection and data usage, authors are expected to elaborate on the rationale of their decision and potential mitigations. Submissions will be evaluated also in terms of the depth of such ethical reflections.
References

[1] J. Whittlestone, R. Nyrup, A. Alexandrova, K. Dihal, and S. Cave. (2019) Ethical and societal implications of algorithms, data, and artificial intelligence: a roadmap for research. London: Nuffield Foundation.

[2] B. Hecht, L. Wilcox, J. P. Bigham, J. Schoning, E. Hoque, J. Ernst, Y. Bisk, L. De Russis, L. Yarosh, B. Anjam, D. Contractor, and C. Wu. (2018) It’s Time to Do Something: Mitigating the Negative Impacts of Computing Through a Change to the Peer Review Process. ACM Future of Computing Blog.

The NeurIPS ethics guidelines were prepared by Samy Bengio, Kate Crawford, Jeanne Fromer, Iason Gabriel, Amanda Levendowski, Deborah Raji and Marc'Aurelio Ranzato, with support and feedback from the NeurIPS 2021 Program Chairs Alina Beygelzimer, Yann Dauphin, Percy Liang, Jenn Wortman Vaughan.

