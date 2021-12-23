Title         : Teacher-Student FrameWork for Zero-Shot Machine Translation
Author        : xxlucas

# Introduction 

TS-FrameWork is machine translation framework which is based on OpenNMT-py.

The main idea about how to implement is coming from **_A Teacher-Student Framework for Zero-Resource Neural Machine Translation_**

Here is the paper :[arXiv:1705.00753]

Although, NMT has achieved state-of-the-art performance on many translation tasks.However, NMT is also plagued by data sparseness.
\cite{zoph2016transfer} indicate that NMT obtains much worse translation quality than a statistical machine translation system.
In response to this problem, it is necessary to construct an effective NMT to cope with the situation of low resource languages.

As a result,Many scholars are working hard to explore new methods of translating languages, when faced with insufficient or even no parallel corpus.
\cite{firat2016multi} present a multi-way,multilingual model with shared attention to achieve zero-resource language pair.Another direction is to develop an universal encoder-decoder architecture.
The simplest and most efficient implementation method is to use target-forcing technique,which is to pretend to the source sentence a tag specifying the target language,both training and testing time(\cite{johnson2017google},\cite{ha2016toward}).
Since the transformer model was proposed, many translation tasks have replaced rnn-based NMT with transformer NMT.
Another direction is to achieve an source-to-target NMT without parallel corpus via a pivot.\cite{cheng2016neural} proposed a method for zero-resource NMT.

Although these approaches prove to be effective,but pivot-based approaches usually need to divide the decoding process into two steps,which is not only more computationally expensive, but also potentially suffers from the error propagation problem(\cite{zhu2013improving}).
\cite{chen2017teacher} proposed an NMT based on knowledge distillation and pre-training.They called source-to-pivot NMT "student" and pivot-to-target NMT "teacher" and they pre-train the "teacher" model and use "teacher" model to guide the learning process of the student model on a source-pivot parallel corpus.
Compared with pivot-based approaches(\cite{cheng2016neural}),their method directly estimates the posterior probability of the source language to the target language.Therefore this strategy not only improves efficiency but also avoids error propagation in decoding.

In this work,we approach low-resource machine translation with so-called pivot-based NMT(\cite{cheng2016neural},\cite{chen2017teacher}).The Transformer approach delivers the best performing multilingual models,with a larger gain over
corresponding bilingual models than observed with RNNs and it delivers the best quality in all considered zero-shot condition and translation directions(\cite{lakew2018comparison}).
Our motivation is based on the above two points and the method of \cite{chen2017teacher} .So our approach is a pivot-based transformer NMT.Not only reduces the calculation cost, but also avoids the second propagation of error. More importantly, the transformer NMT has been proven to perform better than rnn-based NMT.
