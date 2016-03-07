## はじめに ##

**redsvd** は行列分解を解くためのC++ライブラリであり、特異値分解（SVD)、主成分分析(PCA)、固有値分解などをサポートしています．redsvdは非常に大きな行列を扱うことができ、特にtruncated SVD（上位の特異値/固有値に関する分解を行う）に最適化されており、また疎行列を効率的に扱うことができます．例えば、行と列がそれぞれ10万、非零の要素が100万からなる行列に対する上位20位までの特異値分解を1秒未満で行うことができます．

特異値分解アルゴリズムには近年発表された乱択化アルゴリズム[1](1.md)を利用しています．従来の乱択化アルゴリズムは、近似値しか得られなかったのに対し、今回利用しているアルゴリズムは非常に高確率で正しい値を得ることができます．実際の行列に対する結果は実験の章を参照してください．

**headerだけのredsvdをNicolas Tessore氏が作ってくれました． https://github.com/ntessore/redsvd-h**

## ライセンス ##

redsvdはフリーソフトウェアです修正BSDライセンスに基づいて利用できます．

## How to Use ##

現在Linux Ubuntuをサポートしています．（他の環境でも使えると思いますが確認していません）

redsvdではライブラリeigenを利用しています。
eigen3をインストールしてください．[eigen3.0-beta2](http://eigen.tuxfamily.org/index.php?title=3.0_beta)を利用しています．

redsvdではeigen3の行列演算、及びオリジナルのSVDソルバーを利用しています．

次にredsvdの最新のtarballを[Downloads](http://code.google.com/p/redsvd/downloads/list)から入手してください．

最後に次のようにしてインストールしてください。
```
> tar xvjf redsvd-x.x.x.tar.bz2
> cd redsvd-x.x.x
> ./waf configure
> ./waf
sudo ./waf install
```

これにより、プログラムredsvdがインストールされます．

redsvdの使い方は、オプション無しで起動することで確認することができます．
```
>redsvd
usage: redsvd --input=string --output=string [options] ...

redsvd supports the following format types (one line for each row)

[format=dense] (<value>+\n)+
[format=sparse] ((colum_id:value)+\n)+
Example:
>redsvd -i imat -o omat -r 10 -f dense
compuate SVD for a dense matrix in imat and output omat.U omat.V, and omat.S
with the 10 largest eigen values/vectors
>redsvd -i imat -o omat -r 3 -f sparse -m PCA
compuate PCA for a sparse matrix in imat and output omat.PC omat.SCORE
with the 3 largest principal components

options:
  -i, --input     input file (string)
  -o, --output    output file's prefix (string)
  -r, --rank      rank       (int [=10])
  -f, --format    format type (dense|sparse) See example.  (string [=dense])
  -m, --method    method (SVD|PCA|SymEigen) (string [=SVD])
```

密行列の場合はfile1のようなファイルを用意して、それに対する特異値分解を試してみましょう．この例ではfile1.U file1.S file1.Vの三つのファイルが作成されます．
```
> cat file1
 1.0  2.0  3.0  4.0  5.0
-2.0 -1.0  0.0  1.0  2.0
 1.0 -2.0  3.0 -5.0  7.0
> redsvd -i file1 -o file1 -r 2 -f 2
read dense matrix from file1 ... 0.000102997 sec.
SVD for a dense matrix
rows:   3
cols:   5
rank:   2
compute SVD... -4.69685e-05 sec.
write matrix to file1(.U|.S|.V) ... 0.018553 sec.
finished.
> cat file1.U
-0.372291 -0.926217
-0.005434 -0.061765
-0.928100 +0.371897
> cat file1.V
-0.411950 -0.186912
-0.031819 -0.450366
-0.441672 -0.257618
+0.432198 -0.806197
-0.668891 -0.214273
> cat file1.S
+9.176333
+6.647099
```

疎行列の場合にはlibsvdでつかわれている表現方法と同じように各行毎に、0では無い列番号とその値をコロン区切りで書く方法で行列を表します．
```
> 
> cat news20.binary
1:0.016563 3:0.016563 6:0.016563  ...
...

> redsvd -i news20.binary -o news20 -f 1 -r 10
read sparse matrix from news20.binary ... 4.84901 sec.
   rows:        19954
   cols:        1355192
nonzero:        9097916
   rank:        10
compute SVD... 2.52615 sec.
write matrix to news20(.U|.S|.V) ... 5.6056 sec.
finished.
> cat news20.S
+17.973207
+2.556800
+2.460566
+2.135978
+2.022737
+1.931362
+1.927033
+1.853175
+1.770231
+1.764138
```

## 実験結果 ##

グラフ付きの詳細な実験結果については [redsvd\_result.pdf](http://redsvd.googlecode.com/files/redsvd_result_100830.pdf)を参照してください．

これらの結果はインストールした後に、解凍したディレクトリにある
```
> performanceTest.sh
> accuracyTest.sh
```
を実行することで再現できます．


### 実験環境 ###
  * Intel(R) Core(TM)2 Quad CPU Q9550 @ 2.83GHz	8G

### 密行列に対する結果 ###

| n   | m      | rank | time (msec) |
|:----|:-------|:-----|:------------|
| 500 | 100    |   10 | 0.76        |
| 500 | 1000   |   10 | 3.24        |
| 500 | 10000  |   10 | 32.3        |
| 500 | 100000 |   10 | 306.3       |

| n   | m      | rank | time (msec) |
|:----|:-------|:-----|:------------|
| 500 | 100    | 500  | 12.3        |
| 500 | 1000   | 500  | 987.5       |
| 500 | 10000  | 500  | 3850.0      |
| 500 | 100000 | 500  | 32824.3     |

| n     | m     | rank | time (msec) |
|:------|:------|:-----|:------------|
| 100   | 100   |   10 | 0.20        |
| 1000  | 1000  |   10 | 6.34        |
| 10000 | 10000 |   10 | 578         |

| n     | m     | rank | time (msec) |
|:------|:------|:-----|:------------|
| 100   | 100   | 100  | 8.67        |
| 1000  | 1000  | 1000 | 8654        |
| 10000 | 10000 | 1000 | 45001       |

### 疎行列に対する結果 ###

| n      | m      | rank | nonzero ratio (%) |  time (msec) |
|:-------|:-------|:-----|:------------------|:-------------|
| 100    | 100    | 10   | 0.1               | 0.31         |
| 1000   | 1000   | 10   | 0.1               | 1.17         |
| 10000  | 10000  | 10   | 0.1               | 22.5         |
| 100000 | 100000 | 10   | 0.1               | 1679.9       |

| n      | m      | rank | nonzero ratio (%) |  time (msec) |
|:-------|:-------|:-----|:------------------|:-------------|
| 100    | 100    | 10   | 1.0               | 0.16         |
| 1000   | 1000   | 10   | 1.0               | 2.0          |
| 10000  | 10000  | 10   | 1.0               | 124.1        |
| 100000 | 100000 | 10   | 1.0               | 12603.4      |

### Wikipedia英語版に対する潜在的意味解析(文書・単語行列に対するSVD)の結果 ###

rank = 10

| # doc | # word | # total words | time (msec) |
|:------|:-------|:--------------|:------------|
| 3560	 | 27106  | 172823        | 27          |
| 46857 | 147144 | 2418406       | 390         |
| 118110 | 261495 | 6142438       | 1073        |
| 233717 | 402239 | 12026852      | 1993        |

## redsvdの内部 ##

redsvdの主要なコードとその説明は[redsvd.hpp](http://code.google.com/p/redsvd/source/browse/trunk/src/redsvd.hpp)に書かれています．ここではその解説を行ないます．

n行m列の行列Aに対し上位r個の特異値に関する特異値分解をすることを考えてみます． 初めに各成分が平均0、分散1のGaussianからサンプルされたn行r列のGaussian行列Oを作ります．
次に、Y = A<sup>t</sup> Oを求め，Yの各列を正規直交化します． この時AYY<sup>t</sup> ≒ Aが成り立ちます． n行r列のB = AYを求めます． AはYによって列方向に圧縮していると考えられ、Y<sup>t</sup>を書けることで元のAに復元できます．そして扱うのは圧縮した状態であるBです． 次に同様の方法で行方向に情報を圧縮します。 まず、各成分が平均0，分散1のGaussianからサンプルされたr行r列の行列Pをサンプリングし、Z= BPを求めます．そしてZの各列を正規直交化します． B = Z Z<sup>t</sup> Bが成り立ちます． 最後にC = Z<sup>t</sup> Bを求めます． Cはr行r列の行列であり、Z<sup>t</sup>によってBを列方向に圧縮し、Zで復元できると考えられます．最後にCに対してSVDを従来手法で求め、C=USV<sup>t</sup>を得ます． Cは非常に小さい行列なので、この部分は高速に行えます．

A = AYY<sup>t</sup> = BY<sup>t</sup> = ZZ<sup>t</sup>BY<sup>t</sup> = ZCY<sup>t</sup> =ZUSV<sup>t</sup>Y<sup>t</sup>となります． ZU、YVは直交行列であり、それぞれAの左特異ベクトル、右ベクトルに対応します． またSはAの特異値が対角にならんだ対角行列です．

## 謝辞 ##

  * redsvdは [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)を利用しています．
  * redsvdは以下の論文の手法を元に実装されました．但し、オリジナルの方法では一方向のみにサンプリングを行っていますがredsvdでは行・列の両方向でサンプリングを行っています．
  1. "Finding structure with randomness: Stochastic algorithms for constructing approximate matrix decompositions", N. Halko, P.G. Martinsson, J. Tropp, arXiv 0909.4061
  * redsvdは[プリファードインフラストラクチャー](http://preferred.jp/index.html)の20%プロジェクトで作られました．