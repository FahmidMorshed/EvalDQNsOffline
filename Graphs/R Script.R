#!/usr/bin/env Rscript
library(tidyverse)
library(readxl)
library(dplyr)
library(tidyr)
library(car)
library(psych)
library(sjPlot)
library(sjlabelled)
library(sjmisc)
library(ggplot2)
library(FSA)
library(gridExtra)
library(QuantPsyc)
library(ppcor)
library(RColorBrewer)
library(rstatix)
library(ez)
library(lme4)
library(nlme)

H1_grd_dm <- read_excel("Desktop/CSC 722/Project/EvalDQNsOffline/Graphs/H1.xlsx", 
                 sheet = "Combined_grd_dm")
View(H1_grd_dm)
colnames(H1_grd_dm)

H1_grd_dm %>% 
  ggplot() +
  aes(x = `Moving Avg no of iterations`, y = `Avg R`, group = DQN, color = DQN) +
  geom_line(aes(group = DQN)) +
  geom_point() +
  facet_wrap(~Reward)+
  labs(x="X x 1000 iterations", y = "Average reward") + 
  ggtitle("Deterministic Gridworld")+ 
  theme(
    plot.title = element_text(face="bold", hjust = 0.5),
    axis.title.x = element_text(face="bold"),
    axis.title.y = element_text(face="bold"),
    strip.text = element_text(face="bold"),
    panel.background = element_rect(fill = 'white', colour = 'grey'),
    text = element_text(size = 20)
  ) 

anova1<-aov(`Avg R` ~ DQN+Reward, data=H1_grd_dm)
summary(lm(`Avg R` ~ DQN+Reward, data=H1_grd_dm))
Summarize(`Avg R` ~ Reward, data=H1_grd_dm, digits=3)
summary(anova1)

H1_grd_ndm <- read_excel("Desktop/CSC 722/Project/EvalDQNsOffline/Graphs/H1.xlsx", 
                        sheet = "Combined_grd_ndm")
View(H1_grd_ndm)
colnames(H1_grd_ndm)

H1_grd_ndm %>% 
  ggplot() +
  aes(x = `Moving Avg no of iterations`, y = `Avg R`, group = DQN, color = DQN) +
  geom_line(aes(group = DQN)) +
  geom_point() +
  facet_wrap(~Reward)+
  labs(x="X x 1000 iterations", y = "Average reward") + 
  ggtitle("Non-deterministic Gridworld")+ 
  theme(
    plot.title = element_text(face="bold", hjust = 0.5),
    axis.title.x = element_text(face="bold"),
    axis.title.y = element_text(face="bold"),
    strip.text = element_text(face="bold"),
    panel.background = element_rect(fill = 'white', colour = 'grey'),
    text = element_text(size = 20)
  ) 

anova1<-aov(`Avg R` ~ DQN+Reward, data=H1_grd_ndm)
summary(lm(`Avg R` ~ DQN+Reward, data=H1_grd_ndm))
Summarize(`Avg R` ~ Reward, data=H1_grd_ndm, digits=3)
summary(anova1)

H1_cartpole <- read_excel("Desktop/CSC 722/Project/EvalDQNsOffline/Graphs/H1.xlsx", 
                         sheet = "Combined_cartpole")
View(H1_cartpole)
colnames(H1_cartpole)

H1_cartpole %>% 
  ggplot() +
  aes(x = `Moving Avg no of iterations`, y = `Avg R`, group = DQN, color = DQN) +
  geom_line(aes(group = DQN)) +
  geom_point() +
  facet_wrap(~Reward)+
  labs(x="X x 1000 iterations", y = "Average reward") + 
  ggtitle("Cartpole")+ 
  theme(
    plot.title = element_text(face="bold", hjust = 0.5),
    axis.title.x = element_text(face="bold"),
    axis.title.y = element_text(face="bold"),
    strip.text = element_text(face="bold"),
    panel.background = element_rect(fill = 'white', colour = 'grey'),
    text = element_text(size = 20)
  ) 

anova1<-aov(`Avg R` ~ DQN+Reward, data=H1_cartpole)
summary(lm(`Avg R` ~ DQN+Reward, data=H1_cartpole))
Summarize(`Avg R` ~ Reward, data=H1_cartpole, digits=3)
summary(anova1)

H2_grd_dm <- read_excel("Desktop/CSC 722/Project/EvalDQNsOffline/Graphs/H2.xlsx", 
                        sheet = "Gridworld_dm")
View(H2_grd_dm)
colnames(H2_grd_dm)

H2_grd_dm %>% 
  ggplot() +
  aes(x = `Number of iterations`, y = R, group = DQN, color = DQN) +
  geom_line(aes(group = DQN)) +
  geom_point() +
  facet_wrap(~Reward)+
  labs(x="Episode Size", y = "Average reward") + 
  ggtitle("Deterministic Gridworld")+ 
  theme(
    plot.title = element_text(face="bold", hjust = 0.5),
    axis.title.x = element_text(face="bold"),
    axis.title.y = element_text(face="bold"),
    strip.text = element_text(face="bold"),
    panel.background = element_rect(fill = 'white', colour = 'grey'),
    text = element_text(size = 20)
  ) 


H2_grd_ndm <- read_excel("Desktop/CSC 722/Project/EvalDQNsOffline/Graphs/H2.xlsx", 
                        sheet = "Gridworld_ndm")
View(H2_grd_ndm)
colnames(H2_grd_ndm)

H2_grd_ndm %>% 
  ggplot() +
  aes(x = `Number of iterations`, y = R, group = DQN, color = DQN) +
  geom_line(aes(group = DQN)) +
  geom_point() +
  facet_wrap(~Reward)+
  labs(x="Episode Size", y = "Avg of R")


H2_cartpole <- read_excel("Desktop/CSC 722/Project/EvalDQNsOffline/Graphs/H2.xlsx", 
                         sheet = "Cartpole")
View(H2_cartpole)
colnames(H2_cartpole)

H2_cartpole %>% 
  ggplot() +
  aes(x = `Number of iterations`, y = R, group = DQN, color = DQN) +
  geom_line(aes(group = DQN)) +
  geom_point() +
  facet_wrap(~Reward)+
  labs(x="Episode Size", y = "Avg of R")

