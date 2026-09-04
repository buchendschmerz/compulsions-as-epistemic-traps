# プロジェクト仕様書: Compulsions as an Epistemic Trap
### Policy-Induced Non-Identifiability of Control — Rigoux et al. (2024) の拡張

このドキュメントはCodexまたはClaude Codeに渡して実装を進めるための仕様書。フェーズごとにタスクを分割してあるので、上から順に依頼すること。各フェーズの成果物(コード・図・数値)をこのファイルとは別にリポジトリに保存し、進捗をコミットする。

---

## 0. 中心主張(このプロジェクトが証明したいこと)

> Compulsions can become an epistemic trap: one repeats a safety action because one is uncertain about control, but repetition itself removes the counterfactual evidence required to resolve that uncertainty.

制御できるか不確かだから安全行動を反復する。しかし反復そのものが、制御可能性を知るのに必要な反実仮想的証拠を消してしまう。これは単なるOCDモデルの改良ではなく、「安全行動が自己の因果知識を破壊する」という一般原理として提示する。

## 0.1 直感的説明(non-specialist向け。Introductionの導入に使うこと)

**例え**: 鍵を閉めたか不安で、何度も確認しに戻る。
- 素朴な説明:「不安が強いから確認回数が増える」で終わる。
- 本研究が突く問い:「なぜ100回確認しても、101回目も不安なのか」
- 答え: 確認するたびに「もし確認しなかったらどうなっていたか」という比較材料が消えていく。「そもそも鍵は最初から閉まっていた」のか「自分が確認したから閉まった」のかを区別する情報が、確認を繰り返すほど手に入らなくなる。これを数式で証明する。

**既存の2つの説明との違い(3行)**:
- 説明A(Rigoux et al.): 自分の行動は効かないと思い込んでいる(bias寄り)
- 説明B(Sakai et al. 2022): 学習回路(eligibility trace/credit assignment)の異常寄り
- 本研究: 思い込みも学習回路も正常でも、確認という行動パターン自体が答え合わせに必要な情報を自分で消してしまう(policyの構造そのものが罠になる)

**一番大きい示唆**: 制御できているかを知るには、違うやり方を試して結果を比べる経験が必要。安全を確保しようとして同じ行動を繰り返すほど、その安全を確保する力が自分にあるかを知る手段を、自分自身で失っていく、という逆説。

---

## 1. 背景(元論文との差分)

- **元モデル(Rigoux et al. 2024)**: POMDP。状態=clean/dirty、行動=cook/wash/check。予防行為の成功確率 p_SUCCESS は**固定のpoint estimate**。低いp_SUCCESSが強迫行動を説明する。
- **既知の限界(著者自身が明記)**: beliefがどう獲得されるかをモデル化していない。confidenceとbiasを分離するにはBayesian formulationが必要。
- **本研究の拡張点**: p_SUCCESS, p_SOILING(=q)を固定値ではなく**学習される未知パラメータ**にし、「安全行動そのものが、そのパラメータを学習不能にする」という自己維持機構を示す。

---

## 2. 数学的核(最優先で実装・検証すること)

1エピソードで:
- q = 最初から危険な状態にいる確率
- p = 1回の予防行為が危険を除去する確率
- k = 予防行為の反復回数

危険が残る確率: **P(harm | k) = q·(1-p)^k**

agentが毎回同じk回だけ予防してからoutcomeを観測する場合、観測できるのは λ = q·(1-p)^k のみ。→ qとpはnon-identifiable(尤度リッジが生じる)。kを変えた条件を複数経験すれば分離可能になる。

**タスク2-1**: この尤度リッジをsympyまたはnumpyで解析的/数値的に導出し、(q, p)平面上に等尤度線を描くコードを書く。→ Fig.2の元データ。
**タスク2-2**: Fisher information matrixを計算し、determinant(=identifiabilityの指標)がkの取り方でどう変わるかを示す。

---

## 3. リサーチクエスチョン / 仮説(実装時に何を測るかの指針)

- RQ1: 強迫的安全行動は、自分の行為の有効性を推定するための情報を減らすか
- RQ2: 小さな初期の効力への疑念が、行動→情報欠如→疑念という正のフィードバックで強迫状態に移行するか
- RQ3: Response Preventionは「恐怖消去」ではなく「因果的制御可能性の再識別可能化」として説明できるか
- RQ4: 客観的危険低下後も強迫が残るhysteresisは生じるか

- H1 (Safety-behavior censoring): 予防行為を繰り返すagentほどp_SUCCESSのposterior entropyが高いまま維持される
- H2 (Self-maintaining compulsion): 初期の小さなdoubt→wash/check増加→diagnostic evidence減少→doubt未修正→さらにwash/checkの正のフィードバックが成立する
- H3 (Identifiability): 強迫agentではq(base risk)とp(行為効果)がposterior上で強く相関し分離推定できない
- H4 (ERP as causal learning): response preventionにより短期的にはpredicted riskが上がるが、長期的にはparameter uncertaintyが急速に低下する
- H5 (Reassurance paradox): checking/reassuranceはstate uncertainty(今安全か)は下げるが、structural uncertainty(自分の行為に因果効果があるか)はERPほど下げない

---

## 3.5 新規性と示唆(Introduction/Discussionに必ず含めること)

**新規性**
1. 数学的新規性: 反復して安全を確保するpolicy自体が、q(基礎危険率)とp(行為効力)を統計的に識別不能にすることを解析的に証明する。Rigouxモデルはp_SUCCESSを固定値としていたため、この「なぜ疑念が反復でも解消しないか」を原理的に説明できなかった。
2. メカニズムの位置づけの転換: bias説(Rigoux, 効力の過小評価)でも学習回路異常説(Sakai et al. 2022, eligibility trace/credit assignment異常)でもなく、「agentの信念更新も学習能力も正常でも、自分の行為policy自体が学習に必要なデータを消してしまう」という第三の説明を提示する。個体の欠陥ではなく、policyと因果構造の間の構造的な罠だという主張。
3. ERPの再解釈: Response Preventionを「恐怖の消去」ではなく「kを変えてq, pの識別可能性を回復させる因果的介入」として情報理論的に定式化する。

**示唆(Discussionで必ず展開すること)**
- 臨床的示唆1: 症状の重症度を予測するのはp_SUCCESSの平均の低さだけでなく、posteriorの不確実性の高さと、自分のpolicyが生む情報量の低さという分離可能な2量である(H1と対応)。
- 臨床的示唆2(Reassurance paradox, H5): checking/reassuranceは「今安全か」への不安は下げるが「自分の行為に因果効果があるか」への構造的不確実性はERPほど下げない。これは「reassuranceは短期的に効くが長期的には症状を維持する」という臨床的によく知られた現象に、初めて定量的な説明を与える。
- 理論・哲学的示唆: 「自分が世界を制御できると知るには、行為を変えたときに世界がどう変わるかを比較できなければならない」。agencyについての知識は内省だけでは成立せず、反実仮想的アクセスに依存する、という主張。OCD研究を超え、causal inference・意思決定理論・行為の哲学を横断する一般原理として提示する。
- 領域横断的一般化: 同じ構造(自分の安全policyが自分の因果知識を破壊する)は、過剰な医療検査(defensive medicine)、過度な回避行動、金融のリスク回避、アルゴリズムの安全policy設計にも適用できる。Discussionの最後でこの一般化に触れ、「OCDモデルの改良」ではなく「一般理論」として着地させる。

---

## 4. 実装スケジュール(1週間で執筆着手まで圧縮)

前提: Claude CodeとCodexを並行稼働させ、シミュレーションはグリッドベースで軽量(個人PCで完結、MCMC不要)なため、通常数週間かかる作業を1週間に圧縮する。各日の作業は前日の成果物に依存するので順番厳守。

### Day 1: 再現 + 数学的核
- 午前: Rigoux et al. (2024)のPOMDP再現(状態: clean/dirty、行動: cook/wash/check)、Fig.3-5相当の再現図
- 午後: non-identifiability解析証明(タスク2-1, 2-2)、尤度リッジ図(Fig.2候補)
- 成果物: `phase0_replication.py`, `phase1_nonidentifiability.py`

### Day 2-3: Bayesian POMDP拡張
- p_SUCCESS ~ Beta(α_p, β_p), p_SOILING(=q) ~ Beta(α_q, β_q)
- 101×101グリッドでposterior更新実装
- 初期belief条件(正確/軽度悲観/重度悲観)で数百〜数千エピソードのシミュレーション実行、posterior mean・variance・entropyをログ
- **Codex/Claude Codeの相互レビューを必須で挟む**(数値誤差が出やすい箇所)
- 成果物: `phase2_bayesian_pomdp.py` + `results/phase2_*.csv`

### Day 4: 情報量指標・介入比較
- posterior entropy, q-p相関, Fisher information, expected information gain, episode長, wash/check反復回数を算出
- free choice / reassurance / ERP / graded ERPの4条件を比較
- 成果物: `phase3_interventions.py` + `results/phase3_*.csv`

### Day 5: Hysteresis & Phase Diagram
- 危険度低下後の回復速度比較(健常 vs epistemic trap agent)
- phase diagram作成(x=initial uncertainty, y=harm cost)
- 成果物: `phase4_hysteresis_phasediagram.py` + Fig.4, Fig.5データ

### Day 6: 図表確定
- 全図をpublication-quality化(matplotlib、統一スタイル)
- Fig.1〜Fig.7を確定:
  - Fig.1: Rigouxモデル → Bayesian causal-learning modelへの拡張概念図
  - Fig.2: q(1-p)^k の尤度リッジ(non-identifiability)
  - Fig.3: Healthy vs compulsive agentのposterior trajectories
  - Fig.4: uncertainty × harm costのphase diagram
  - Fig.5: hysteresis(閾値低下後も症状残存)
  - Fig.6: ERP vs checking/reassuranceのposterior entropy低下比較
  - Fig.7: human experimentへの反証可能な予測(表または概念図)
- 成果物: `figures/`一式

### Day 7: 原稿ドラフト執筆
- LaTeXドラフト作成(Introduction / Model / Results / Discussion)
- **Introductionには0.1の直感的説明と3.5の新規性を必ず組み込む**
- **Discussionには3.5の示唆(臨床的示唆1・2、理論的示唆、領域横断的一般化)を必ず展開する**
- 成果物: `paper_draft.tex` + `figures/`

この後の指導教員レビュー・査読対応は別スケジュール(このファイルの対象外)。

---

## 5. コーディング上の指示(Codex/Claude Code共通)

- 言語: Python(NumPy必須、可能ならJAXでgrid posterior更新を高速化)
- グリッドベースのBayesian更新を優先し、MCMCは使わない(計算コストと再現性の両面で有利)
- 乱数シードを固定し、全シミュレーションを再現可能にする
- 各フェーズごとに独立したスクリプト+テストを作成し、`results/`にnumpy/csv形式で中間データを保存する
- **相互レビュー運用**: 一方のツール(Codex or Claude Code)が実装したコードを、もう一方に読ませてバグ・数値誤差・統計的妥当性をレビューさせる。特にPhase2のposterior更新とPhase2-3の情報量計算は数値誤差が出やすいので必須。

---

## 6. 完了の判断基準(Definition of Done)

- Phase1の解析的結果とPhase2のシミュレーション結果が定性的に一致すること(non-identifiabilityがシミュレーションでも再現される)
- H1〜H5の各仮説について支持/不支持を数値的に判定できるデータが揃っていること
- 全Figureがドラフトに挿入可能な状態(publication-quality, ラベル・凡例完備)であること
- LaTeXドラフトが Introduction〜Discussion まで一通り埋まっていること

---

## 7. 参考: 元論文情報

- Rigoux, L. et al. (2024). POMDPベースのOCDモデル(状態: clean/dirty、行動: cook/wash/check)。p_SUCCESS過小評価がcompulsionを説明。
- 関連研究: Sakai et al. (2022) — OCDをreinforcement/punishment間のeligibility trace異常としてモデル化(本研究とは主張の軸が異なるため、credit assignment deficitではなくpolicy-induced non-identifiabilityに主張を置くこと)
