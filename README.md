# Music-mixed Speech Recognition   
https://arxiv.org/abs/2008.12048 のデモです。
## Requirements   

> librosa 0.7.2   
  numpy 1.18.1   
  torch 1.5.0   


## 説明
> **src/** の下にソースコードがあります。   
  **src.\*** ファイルはタスクの実行スクリプトファイルです。sepは分離タスク、evalは認識タスクです。   
  asr.cleanはクリーンデータで学習したASRモデル、asr.mixtureは混合音源で学習したASRモデルです。   
  all.cleanはクリーンデータで学習したASRもでるをつかったジョイントもでる、all.mixtureは混合音源で学習したASRモデルをつかったジョイントモデルです。   
  **result/** には分離した結果があって、**log/** には認識した結果があります。   
  
  
## リアルデータサンプル
> **wav/** の下にサンプルのリアルデータがあります。wavファイルは16kのsamplerateです。   
  このサンプルは今回直接録音したもので、Pythonで学ぶ音源分離（戸上真人著）の序章の最初の文章2つを読み上げたものです。   
  **wav/clean/** にはクリーンで録音したもので、**wav/mixed_speech/** には録音時にスピーカで音楽を再生しながらとったものです。音楽は著作権フリーのものを使いました。(https://maoudamashii.jokersounds.com/) (http://drums.kirakira-soundeffect.com/)   
  読み上げは京都大学情報学研究科の松浦孝平が担当しました。  
  
  
## 学習済みデータ
> https://drive.google.com/drive/folders/1psIeO9VKVsB0Cd1SNVvVAp3U7bDWBsmi?usp=sharing に学習済みデータがあります。   
  
## 音声認識
> bpe.idでサブワード系列の定義を示しています。sosは0, eosは1です。9515個のサブワードがあります。
