---
# You can also start simply with 'default'
theme: seriph
# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
background: https://plus.unsplash.com/premium_photo-1706838708757-90894d747ada?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxjb2xsZWN0aW9uLXBhZ2V8N3w5NDczNDU2Nnx8ZW58MHx8fHx8
# some information about your slides (markdown enabled)
title: Hyperparameter Hippies
info: |
  ## Week 3
# apply unocss classes to the current slide
class: text-center
# https://sli.dev/features/drawing
drawings:
  persist: false
# slide transition: https://sli.dev/guide/animations.html#slide-transitions
transition: slide-left
# enable MDC Syntax: https://sli.dev/features/mdc
mdc: true
# open graph
# seoMeta:
#  ogImage: https://cover.sli.dev
---

# Hyperparameter Hippies

## Week 3

Anton Dergunov, Dan Goss, Ben Liong, Tyrone Nicholas

---
transition: fade-out
layout: two-cols-header
---
# Lies and the Lying Liars who tell them
::left::

<ul>
<v-click at="1"><li>Profile and tune to make training faster</li></v-click>
<v-click at="3"><li>Refine your model to improve inference</li></v-click>
<v-click at="5"><li>Upgrade to latest libraries</li></v-click>
<v-click at="7"><li>Write parallel code in parallel</li></v-click>
<v-click at="9"><li><code>torch.compile</code> for speed</li></v-click>
<v-click at="11"><li>Academic papers are a clear blueprint to write code</li></v-click>
</ul>
::right::
<ul>
<v-click at="2"><li>Try a new GPU</li></v-click>
<v-click at="4"><li>Modify your training data</li></v-click>
<v-click at="6"><li>Use the preinstalled pytorch</li></v-click>
<v-click at="8"><li>Write it in serial</li></v-click>
<v-click at="10"><li>No</li></v-click>
<v-click at="12"><li>No</li></v-click>
</ul>
---
transition: slide-up
level: 2
---

# Demo

<a href="http://localhost:8501" target="_blank">First Demo</a>


<PoweredBySlidev mt-10 />
