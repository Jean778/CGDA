# ============================================================
# Repository root (REPO) resolves automatically in Rscript and RStudio.
# Override with env var, e.g.  REPO=<repo-root>  Rscript figures/plot_x.R
# ============================================================
if (Sys.getenv("REPO") != "") {
  REPO <- Sys.getenv("REPO")
} else if (interactive()) {
  if (requireNamespace("rstudioapi", quietly = TRUE)) {
    REPO <- dirname(dirname(rstudioapi::getSourceEditorContext()$path))
  } else {
    REPO <- getwd()
  }
} else {
  .cgda_args <- commandArgs(trailingOnly = FALSE)
  .cgda_file <- sub("^--file=", "", .cgda_args[grepl("^--file=", .cgda_args)])
  REPO <- if (length(.cgda_file) > 0) dirname(dirname(.cgda_file)) else getwd()
}
library(ggplot2)
library(dplyr)
library(tidyr)

path <- file.path(REPO, "main_results/")

# ============================================================
# 1.   
# ============================================================

budget_use <- 0.10
seed_use <- 42
fold_use <- 5
pair_use <- "4-6"
method_use <- "cg_sup"
rho_use <- 0.50

methods <- c("base", "cal_sup", "cg_sup")

method_cols <- c(
  base = "#D83C3C",
  cal_sup = "#E6A15C",
  cg_sup = "#258C88"
)

method_labels <- c(
  base = "Source Only",
  cal_sup = "Cal-Sup",
  cg_sup = "CGDA"
)

theme_pub <- theme_bw(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    strip.text = element_text(face = "bold"),
    legend.position = "top",
    legend.title = element_blank(),
    plot.caption = element_text(hjust = 0, size = 9)
  )

# ============================================================
# 2.     
# ============================================================

a <- read.csv(paste0(path, "panel_44a_oscr_cosine.csv"))
b <- read.csv(paste0(path, "panel_44b_contamination_cosine.csv"))
s <- read.csv(paste0(path, "panel_44_sample_scores_cosine.csv"))
t <- read.csv(paste0(path, "panel_44_thresholds_cosine.csv"))

s <- s %>%
  filter(
    abs(budget - budget_use) < 1e-8,
    method %in% methods
  )

t <- t %>%
  filter(
    abs(budget - budget_use) < 1e-8,
    method %in% methods
  )

# a b  budget          budget_use
a <- a %>% filter(method %in% methods)
b <- b %>% filter(method %in% methods, strategy %in% c('full','tail')) %>% distinct()

complete_runs <- s %>% distinct(unknown_pair,seed,fold,method) %>%
 group_by(unknown_pair,seed,fold) %>% filter(n_distinct(method)==length(methods)) %>% ungroup()
excluded_runs <- s %>% distinct(unknown_pair,seed,fold,method) %>% anti_join(complete_runs,by=c('unknown_pair','seed','fold','method'))
if(nrow(excluded_runs)>0) {message('Incomplete paired runs excluded; inspect excluded_runs.'); print(excluded_runs)}
s <- semi_join(s,complete_runs,by=c('unknown_pair','seed','fold','method'))
keys <- c(
  "protocol", "direction", "budget", "seed",
  "fold", "method", "unknown_pair"
)

stopifnot(
  nrow(s) > 0,
  all(is.finite(s$distance_score)),
  all(s$is_unknown %in% c(0, 1)),
  !anyNA(s$y_true),
  !anyNA(s$y_pred),
  !anyDuplicated(s[c(keys, "sample_id")])
)

# ============================================================
# 3. OSCR      
# ============================================================

# Rebuild budget-specific curves and pollution metrics from sample scores.
# Summary exports omit budget and may mix incompatible runs.
a <- s %>% group_by(across(all_of(keys))) %>% group_modify(~{
 z <- .x; nk <- sum(z$is_unknown==0); nu <- sum(z$is_unknown==1)
 stopifnot(nk>0,nu>0)
 v <- z %>% group_by(distance_score) %>% summarise(
 u=sum(is_unknown==1), k=sum(is_unknown==0 & y_pred==y_true),.groups='drop') %>% arrange(distance_score)
 tibble(unknown_fpr=c(0,cumsum(v$u)/nu),known_ccr=c(0,cumsum(v$k)/nk))
}) %>% ungroup() %>% rename(setting=unknown_pair)
b <- s %>% group_by(across(all_of(keys))) %>% group_modify(~{
 z <- .x; q <- semi_join(t,.y,by=keys) %>% filter(grepl('_(full|tail)_q90$',threshold_method))
 stopifnot(nrow(q)>0,!anyDuplicated(q$threshold_method))
 k<-z$is_unknown==0;u<-z$is_unknown==1
 tibble(strategy=ifelse(grepl('_full_',q$threshold_method),'full','tail'),rho=q$contamination_ratio,
 known_correct_acceptance=vapply(q$threshold,function(tau)mean(z$distance_score[k]<=tau & z$y_pred[k]==z$y_true[k]),numeric(1)),
 unknown_rejection=vapply(q$threshold,function(tau)mean(z$distance_score[u]>tau),numeric(1)))
}) %>% ungroup() %>% rename(setting=unknown_pair)
a_metrics <- a %>%
  group_by(setting, method, seed, fold) %>%
  group_modify(~ {
    
    z <- .x %>%
      arrange(unknown_fpr, known_ccr)
    
    stopifnot(
      all(is.finite(z$unknown_fpr)),
      all(is.finite(z$known_ccr)),
      all(z$unknown_fpr >= 0 & z$unknown_fpr <= 1),
      all(z$known_ccr >= 0 & z$known_ccr <= 1),
      min(z$unknown_fpr) == 0,
      max(z$unknown_fpr) == 1
    )
    
    tibble(
      OSCR_AUC = sum(
        diff(z$unknown_fpr) *
          (
            head(z$known_ccr, -1) +
              tail(z$known_ccr, -1)
          ) / 2
      ),
      
      CCR_UAR05 = max(
        z$known_ccr[z$unknown_fpr <= 0.05]
      )
    )
  }) %>%
  ungroup()

# ============================================================
# 4. OSCR              
# ============================================================

grid_x <- seq(0, 1, length.out = 501)

a_grid <- a %>%
  group_by(setting, method, seed, fold) %>%
  group_modify(~ {
    
    z <- .x %>%
      group_by(unknown_fpr) %>%
      summarise(
        y = max(known_ccr),
        .groups = "drop"
      ) %>%
      arrange(unknown_fpr)
    
    index <- findInterval(
      grid_x,
      z$unknown_fpr
    )
    
    tibble(
      UAR = grid_x,
      CCR = z$y[index]
    )
  }) %>%
  ungroup()

#       CCR          
a_sum <- a_grid %>%
  group_by(setting, method, UAR) %>%
  summarise(
    mean_CCR = mean(CCR),
    sd_CCR = sd(CCR),
    .groups = "drop"
  )

stopifnot(
  all(is.finite(a_sum$mean_CCR)),
  all(is.finite(a_sum$sd_CCR))
)

# ============================================================
# 5.                  
# ============================================================

t_known <- t %>%
  filter(
    threshold_method == "known_reference_q90"
  ) %>%
  select(all_of(keys), threshold)

stopifnot(
  !anyDuplicated(t_known[keys]),
  all(is.finite(t_known$threshold))
)

matched <- left_join(
  s,
  t_known,
  by = keys
)

stopifnot(
  !anyNA(matched$threshold),
  nrow(matched) == nrow(s)
)

run_stats <- matched %>%
  group_by(across(all_of(keys))) %>%
  group_modify(~ {
    
    z <- .x
    k <- z$is_unknown == 0
    u <- z$is_unknown == 1
    tau <- unique(z$threshold)
    
    stopifnot(
      length(tau) == 1,
      any(k),
      any(u)
    )
    
    ranks <- rank(
      z$distance_score,
      ties.method = "average"
    )
    
    tibble(
      known_accept = mean(
        z$distance_score[k] <= tau
      ),
      
      CCR = mean(
        z$distance_score[k] <= tau &
          z$y_pred[k] == z$y_true[k]
      ),
      
      UAR = mean(
        z$distance_score[u] <= tau
      ),
      
      unknown_reject = mean(
        z$distance_score[u] > tau
      ),
      
      AUROC = (
        sum(ranks[u]) -
          sum(u) * (sum(u) + 1) / 2
      ) / (sum(u) * sum(k))
    )
  }) %>%
  ungroup() %>%
  rename(setting = unknown_pair)

run_id <- c(
  "setting", "method", "seed", "fold"
)

stopifnot(
  !anyDuplicated(run_stats[run_id]),
  
  nrow(
    anti_join(a_metrics, run_stats, by = run_id)
  ) == 0,
  
  nrow(
    anti_join(run_stats, a_metrics, by = run_id)
  ) == 0
)

work_points <- run_stats %>%
  group_by(setting, method) %>%
  summarise(
    UAR = mean(UAR),
    CCR = mean(CCR),
    .groups = "drop"
  )

# ============================================================
# 6.      
# ============================================================

metrics_table <- a_metrics %>%
  left_join(
    run_stats %>%
      select(
        all_of(run_id),
        AUROC,
        known_accept,
        CCR,
        unknown_reject
      ),
    by = run_id
  ) %>%
  group_by(setting, method) %>%
  summarise(
    n_runs = n(),
    
    across(
      c(
        OSCR_AUC,
        CCR_UAR05,
        AUROC,
        known_accept,
        CCR,
        unknown_reject
      ),
      list(mean = mean, sd = sd),
      .names = "{.col}_{.fn}"
    ),
    
    .groups = "drop"
  )

# ============================================================
# 7. OSCR       
# ============================================================

p_oscr_base <- ggplot(
  a_sum,
  aes(
    x = UAR,
    y = mean_CCR,
    colour = method,
    fill = method,
    group = method
  )
) +
  geom_ribbon(
    aes(
      ymin = pmax(0, mean_CCR - sd_CCR),
      ymax = pmin(1, mean_CCR + sd_CCR)
    ),
    alpha = 0.10,
    colour = NA
  ) +
  geom_line(linewidth = 0.9) +
  geom_vline(
    xintercept = 0.05,
    linetype = "dotted",
    colour = "grey50",
    linewidth = 0.4
  ) +
  geom_point(
    data = work_points,
    aes(
      x = UAR,
      y = CCR,
      colour = method
    ),
    inherit.aes = FALSE,
    shape = 18,
    size = 3.2
  ) +
  scale_colour_manual(
    values = method_cols,
    breaks = methods,
    labels = unname(method_labels[methods])
  ) +
  scale_fill_manual(
    values = method_cols,
    guide = "none"
  ) +
  facet_wrap(
    ~setting,
    nrow = 1,
    labeller = labeller(
      setting = function(x) {
        paste("Unknown:", x)
      }
    )
  ) +
  labs(
    x = "Unknown acceptance rate",
    y = "Known correct acceptance rate",
    caption = paste0(
      "Curves: run means; shading: +/-1 between-run SD.\n",
      "Diamonds: mean working points from run-specific known-reference Q90.\n",
      "Mean working points need not lie on the mean curves."
    )
  ) +
  theme_pub

#     
p_oscr <- p_oscr_base +
  scale_x_continuous(
    breaks = seq(0, 1.1, 0.2)
  ) +
  coord_cartesian(
    xlim = c(-0.01, 1),
    ylim = c(0, 1.1),
    expand = FALSE
  )

#        
p_oscr_zoom <- p_oscr_base +
  scale_x_continuous(
    breaks = seq(0, 0.10, 0.025)
  ) +
  coord_cartesian(
    xlim = c(0, 0.10),
    ylim = c(0, 1),
    expand = FALSE
  )

# ============================================================
# 8.      
# ============================================================

stopifnot(
  !anyDuplicated(
    b[
      c(
        "setting", "method", "strategy",
        "rho", "seed", "fold"
      )
    ]
  ),
  all(is.finite(b$known_correct_acceptance)),
  all(is.finite(b$unknown_rejection))
)

#               
coverage <- b %>%
  group_by(
    setting, method, strategy, rho
  ) %>%
  summarise(
    runs = paste(
      sort(paste(seed, fold, sep = "_")),
      collapse = ";"
    ),
    .groups = "drop"
  ) %>%
  group_by(setting) %>%
  summarise(
    n_sets = n_distinct(runs),
    .groups = "drop"
  )

stopifnot(all(coverage$n_sets == 1))

b_sum <- b %>%
  pivot_longer(
    c(
      known_correct_acceptance,
      unknown_rejection
    ),
    names_to = "metric",
    values_to = "value"
  ) %>%
  group_by(
    setting, method, strategy, rho, metric
  ) %>%
  summarise(
    avg = mean(value),
    SD = sd(value),
    .groups = "drop"
  )

pd <- position_dodge(width = 0.035)

p_contam <- ggplot(
  b_sum,
  aes(
    x = rho,
    y = avg,
    colour = method,
    linetype = strategy,
    group = interaction(method, strategy)
  )
) +
  geom_errorbar(
    aes(
      ymin = pmax(0, avg - SD),
      ymax = pmin(1, avg + SD)
    ),
    position = pd,
    width = 0.012,
    linewidth = 0.35,
    alpha = 0.35
  ) +
  geom_line(
    position = pd,
    linewidth = 0.8
  ) +
  geom_point(
    position = pd,
    size = 1.8
  ) +
  scale_colour_manual(
    values = method_cols,
    breaks = methods,
    labels = unname(method_labels[methods])
  ) +
  scale_linetype_manual(
    values = c(
      full = "solid",
      tail = "dashed"
    ),
    labels = c(
      full = "Full pool",
      tail = "Low-distance subset"
    )
  ) +
  scale_x_continuous(
    breaks = sort(unique(b$rho))
  ) +
  coord_cartesian(
    ylim = c(0, 1)
  ) +
  facet_grid(
    metric ~ setting,
    labeller = labeller(
      metric = c(
        known_correct_acceptance = "Known correct acceptance",
        unknown_rejection = "Unknown rejection"
      ),
      setting = function(x) {
        paste("Unknown:", x)
      }
    )
  ) +
  labs(
    x = "Unknown fraction in reference pool",
    y = "Proportion",
    caption = paste(
      "Points: run means.",
      "Error bars: +/-1 between-run SD."
    )
  ) +
  theme_pub

# ============================================================
# 9.        ECDF       
# ============================================================

sc <- s %>%
  filter(
    method == method_use,
    unknown_pair == pair_use,
    seed == seed_use,
    fold == fold_use
  )

rules <- paste0(
  "mixed",
  sprintf("%02d", round(100 * rho_use)),
  c("_full_q90", "_tail_q90")
)

tc <- t %>%
  filter(
    method == method_use,
    unknown_pair == pair_use,
    seed == seed_use,
    fold == fold_use,
    threshold_method %in% rules
  ) %>%
  mutate(
    strategy = ifelse(
      grepl("_full_", threshold_method),
      "Full pool",
      "Low-distance subset"
    )
  )

stopifnot(
  nrow(sc) > 0,
  nrow(unique(sc[keys])) == 1,
  nrow(tc) == 2,
  !anyDuplicated(tc$strategy),
  all(is.finite(tc$threshold)),
  nrow(
    anti_join(tc, unique(sc[keys]), by = keys)
  ) == 0,
  any(sc$is_unknown == 0),
  any(sc$is_unknown == 1)
)

sc$group <- ifelse(
  sc$is_unknown == 1,
  "Unknown",
  "Known"
)

#                    
tc$known_accept <- sapply(
  tc$threshold,
  function(q) {
    mean(
      sc$distance_score[sc$is_unknown == 0] <= q
    )
  }
)

tc$CCR <- sapply(
  tc$threshold,
  function(q) {
    k <- sc$is_unknown == 0
    
    mean(
      sc$distance_score[k] <= q &
        sc$y_pred[k] == sc$y_true[k]
    )
  }
)

tc$unknown_reject <- sapply(
  tc$threshold,
  function(q) {
    mean(
      sc$distance_score[sc$is_unknown == 1] > q
    )
  }
)

#     log10              
#                 
use_log <- all(sc$distance_score > 0) &&
  all(tc$threshold > 0)

sc$x <- if (use_log) {
  log10(sc$distance_score)
} else {
  sc$distance_score
}

tc$x <- if (use_log) {
  log10(tc$threshold)
} else {
  tc$threshold
}

#              ECDF
ec <- sc %>%
  count(group, x) %>%
  group_by(group) %>%
  arrange(x, .by_group = TRUE) %>%
  mutate(
    F = cumsum(n) / sum(n)
  ) %>%
  ungroup()

#       
xr <- range(c(sc$x, tc$x))
pad <- max(diff(xr), 1) * 0.04

ec <- bind_rows(
  ec %>%
    select(group, x, F),
  
  data.frame(
    group = c("Known", "Unknown"),
    x = xr[1] - pad,
    F = 0
  ),
  
  data.frame(
    group = c("Known", "Unknown"),
    x = xr[2] + pad,
    F = 1
  )
) %>%
  arrange(group, x)

stats_caption <- paste(
  sprintf(
    paste0(
      "%s: tau=%.4g; known accept=%.1f%%; ",
      "CCR=%.1f%%; unknown reject=%.1f%%"
    ),
    tc$strategy,
    tc$threshold,
    100 * tc$known_accept,
    100 * tc$CCR,
    100 * tc$unknown_reject
  ),
  collapse = "\n"
)

p_dist <- ggplot(
  ec,
  aes(x = x, y = F, colour = group)
) +
  geom_step(
    linewidth = 0.9
  ) +
  geom_vline(
    data = tc,
    aes(
      xintercept = x,
      linetype = strategy
    ),
    colour = "black",
    linewidth = 0.7
  ) +
  scale_colour_manual(
    values = c(
      Known = "#258C88",
      Unknown = "#D83C3C"
    )
  ) +
  scale_linetype_manual(
    values = c(
      "Full pool" = "solid",
      "Low-distance subset" = "dashed"
    )
  ) +
  scale_y_continuous(
    breaks = seq(0, 1, 0.25)
  ) +
  labs(
    x = if (use_log) {
      "log10(Distance score)"
    } else {
      "Distance score"
    },
    
    y = "Cumulative proportion",
    
    subtitle = sprintf(
      paste0(
        "%s | Unknown %s | Seed %d | Fold %d | ",
        "Contamination %.0f%%"
      ),
      method_use,
      pair_use,
      seed_use,
      fold_use,
      100 * rho_use
    ),
    
    caption = paste0(
      stats_caption,
      "\nTest n: known=",
      sum(sc$is_unknown == 0),
      "; unknown=",
      sum(sc$is_unknown == 1),
      ". Thresholds come from the separate reference pool."
    )
  ) +
  theme_pub

# ============================================================

# Visual styling only; preceding run-level calculations are retained.
library(patchwork)
method_cols <- c(base='#354F80',cal_sup='#DE7548',cg_sup='#258C88')
theme_clean <- theme_classic(base_size=11)+theme(
 axis.line=element_line(linewidth=.9,colour='black'),
 axis.ticks=element_line(linewidth=.9,colour='black'),
 axis.text=element_text(size=12,colour='black', face = 'bold'),
 axis.title=element_text(size=12,colour='black', face = 'bold'),
 plot.title=element_text(size=11,face='bold',hjust=.5),
 legend.title=element_blank(),legend.position='bottom',
 legend.text=element_text(size=14, face = 'bold'),
 plot.margin=margin(8,12,8,8))
settings <- c('1-2','3-5','4-6')

# OSCR: each setting has its own axes; colored in-panel method/AUC labels.
oscr_panel <- function(up,zoom=FALSE){
 z<-filter(a_sum,setting==up);w<-filter(work_points,setting==up)
 ann<-filter(metrics_table,setting==up)
 ann<-ann[match(methods,ann$method),]
 ann$yy<-c(.27,.18,.09)
 ann$txt<-sprintf('%s  AUC: %.3f',unname(method_labels[ann$method]),ann$OSCR_AUC_mean)
 xmax<-if(zoom).10 else 1
 ggplot(z,aes(UAR,mean_CCR,colour=method,fill=method,group=method))+
 geom_ribbon(aes(ymin=pmax(0,mean_CCR-sd_CCR),ymax=pmin(1,mean_CCR+sd_CCR)),alpha=.08,colour=NA)+
 geom_line(linewidth=.7)+
 geom_vline(xintercept=.05,linetype='dotted',colour='grey60',linewidth=.7)+
 geom_point(data=w,aes(UAR,CCR,colour=method),inherit.aes=FALSE,shape=18,size=4.5)+
 geom_label(data=ann,aes(x=.96*xmax,y=yy,label=txt,colour=method),inherit.aes=FALSE,hjust=1,size=2.8,fill='white',linewidth=0,label.padding=grid::unit(.1,'lines'))+
 scale_colour_manual(values=method_cols)+scale_fill_manual(values=method_cols)+
 scale_x_continuous(breaks=if(zoom)seq(0,.1,.025) else seq(0,1,.25))+
 scale_y_continuous(breaks=c(0,.5,1))+
 coord_cartesian(xlim=c(0,xmax),ylim=c(0,1.02))+
 labs(title=paste0('Unknown:',up),x='Unknown Acceptance Rate',y='Known Correct Acceptance Rate')+
 theme_clean+theme(legend.position='none')
}
# oscr_note<-'Shading: +/-1 between-run SD. Diamonds: mean known-reference Q90 working points.\nAUC labels describe the full curve, including in the zoomed view; working points outside the zoom are not shown.'
p_oscr<-wrap_plots(lapply(settings,oscr_panel),nrow=1)
p_oscr_zoom<-wrap_plots(lapply(settings,function(up)oscr_panel(up,TRUE)),nrow=1)+plot_annotation(caption=oscr_note)

# Contamination: six curves require a shared legend rather than in-panel text.
contam_panel<-function(up,metric_name){
 z<-b_sum[b_sum$setting==up & b_sum$metric==metric_name,]
 ggplot(z,aes(rho,avg,colour=method,linetype=strategy,group=interaction(method,strategy)))+
 geom_errorbar(aes(ymin=pmax(0,avg-SD),ymax=pmin(1,avg+SD)),width=.03,linewidth=1.2,alpha=.28)+
 geom_line(linewidth=1.2)+geom_point(aes(shape=strategy),size=3)+
 scale_colour_manual(values=method_cols,breaks=methods,labels=unname(method_labels[methods]))+
 scale_linetype_manual(values=c(full='solid',tail='dashed'),breaks=c('full','tail'),labels=c('Full pool','Low-distance subset'))+
 scale_shape_manual(values=c(full=16,tail=17),breaks=c('full','tail'),labels=c('Full pool','Low-distance subset'))+
 scale_x_continuous(breaks=sort(unique(b$rho)))+scale_y_continuous(breaks=c(0,.5,1))+
 coord_cartesian(ylim=c(0,1.02))+
 labs(title=paste0('Unknown:',up),x='Unknown Fraction in Reference Pool',y=if(metric_name=='known_correct_acceptance')'Known Correct Acceptance' else 'Unknown Rejection')+
 theme_clean
}
p_contam<-wrap_plots(c(
 lapply(settings,function(up)contam_panel(up,'known_correct_acceptance')),
 lapply(settings,function(up)contam_panel(up,'unknown_rejection'))),ncol=3,guides='collect')+
 plot_annotation(caption='') &theme(legend.position='bottom')

# ECDF: identical test curves, separate full/tail thresholds and rates.
# Use linear cosine distance; zeros are retained.
ec_style<-sc %>% count(group,distance_score) %>% group_by(group) %>% arrange(distance_score,.by_group=TRUE) %>% mutate(F=cumsum(n)/sum(n)) %>% ungroup()
xhi<-max(c(sc$distance_score,tc$threshold))*1.04
stopifnot(is.finite(xhi),xhi>0,all(sc$distance_score>=0))
ec_style<-bind_rows(ec_style %>% select(group,distance_score,F),
 data.frame(group=c('Known','Unknown'),distance_score=-xhi*1e-8,F=0),
 data.frame(group=c('Known','Unknown'),distance_score=xhi,F=1)) %>% arrange(group,distance_score)
dist_panel<-function(strategy_name){
 q<-tc[tc$strategy==strategy_name,];stopifnot(nrow(q)==1)
 ann<-data.frame(group=c('Known','Unknown'),yy=c(.50,.39),label=c(
 sprintf('Known(n=%d)  accept:%.1f%%',sum(sc$is_unknown==0),100*q$known_accept),
 sprintf('Unknown(n=%d)  reject:%.1f%%',sum(sc$is_unknown==1),100*q$unknown_reject)))
 ggplot(ec_style,aes(distance_score,F,colour=group))+
 geom_step(linewidth=1.0)+
 annotate('segment',x=q$threshold,xend=q$threshold,y=0,yend=1,linetype='dashed',colour='grey40',linewidth=1.0)+
 annotate('text',x=.98*xhi,y=1.06,label=sprintf('tau = %.3g',q$threshold),hjust=1,size=4)+
 geom_label(data=ann,aes(x=.78*xhi,y=yy,label=label,colour=group),inherit.aes=FALSE,hjust=1,size=3.5,fill='white',linewidth=0,label.padding=grid::unit(.1,'lines'))+
 annotate('text',x=.78*xhi,y=.28,label=sprintf('Known correct accept: %.1f%%',100*q$CCR),hjust=1,size=3.5,colour='grey30')+
 scale_colour_manual(values=c(Known='#354F80',Unknown='#DE7548'))+
 scale_x_continuous(breaks=scales::breaks_pretty(n=4))+scale_y_continuous(breaks=c(0,.5,1))+
 coord_cartesian(xlim=c(0,xhi),ylim=c(0,1.12))+
 labs(title=strategy_name,x='Cosine Distance Sore',y='Cumulative Proportion')+
 theme_clean+theme(legend.position='none')
}
p_dist<-wrap_plots(lapply(c('Full pool','Low-distance subset'),dist_panel),nrow=1)+
 plot_annotation(title=sprintf('%s | Unknown:%s',unname(method_labels[method_use]),pair_use),subtitle=sprintf('Budget %.0f%% | Seed %d | Fold %d | Contamination %.0f%%',100*budget_use,seed_use,fold_use,100*rho_use))

print(p_oscr)
# print(p_oscr_zoom)
print(p_contam)
print(p_dist)
# ggsave(file.path(path,'oscr_styled.png'),p_oscr,width=11,height=4.2,dpi=300)
# ggsave(file.path(path,'contamination_styled.png'),p_contam,width=11,height=7,dpi=300)
# ggsave(file.path(path,'ecdf_styled.png'),p_dist,width=9,height=4.8,dpi=300)
