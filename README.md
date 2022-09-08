# Debussy's oeuvre for piano

Delicious dataset

Command for creating pitch class vectors: `dimcat pcvs -o pcvs -w 0.5 -p pc -q 1 --fillna 0.0 --round 5` (dimcat > 0.2.0) where 

* `-o pcvs` stands for the output directory,
* `-w 0.5` means weighting grace notes by half of their durations,
* `-p pc` using pitch classes 0..11
* `-q 1` means slices of length 1 quarter,
* `--fillna 0.0` fills empty fields (=non-occurrent pitch classes) with 0.0
* `--round 5` rounds the output to 5 (maximum available precision).


# Overview
|         file_name          |measures|labels|standard|annotators|reviewers|
|----------------------------|-------:|-----:|--------|----------|---------|
|l000_etude                  |      71|     0|        |          |         |
|l000_soirs                  |      23|     0|        |          |         |
|l009_danse                  |      92|     0|        |          |         |
|l066-01_arabesques_premiere |     107|     0|        |          |         |
|l066-02_arabesques_deuxieme |     110|     0|        |          |         |
|l067_mazurka                |     138|     0|        |          |         |
|l068_reverie                |     101|     0|        |          |         |
|l069_tarentelle             |     333|     0|        |          |         |
|l070_ballade                |     105|     0|        |          |         |
|l071_valse                  |     151|     0|        |          |         |
|l075-01_suite_prelude       |      89|     0|        |          |         |
|l075-02_suite_menuet        |     104|     0|        |          |         |
|l075-03_suite_clair         |      72|     0|        |          |         |
|l075-04_suite_passepied     |     156|     0|        |          |         |
|l082_nocturne               |      77|     0|        |          |         |
|l087-01_images_lent         |      57|     0|        |          |         |
|l087-02_images_souvenir     |      72|     0|        |          |         |
|l087-03_images_quelques     |     186|     0|        |          |         |
|l095-01_pour_prelude        |     163|     0|        |          |         |
|l095-02_pour_sarabande      |      72|     0|        |          |         |
|l095-03_pour_toccata        |     266|     0|        |          |         |
|l099_cahier                 |      55|     0|        |          |         |
|l100-01_estampes_pagode     |      98|     0|        |          |         |
|l100-02_estampes_soiree     |     136|     0|        |          |         |
|l100-03_estampes_jardins    |     157|     0|        |          |         |
|l105_masques                |     381|     0|        |          |         |
|l106_isle                   |     255|     0|        |          |         |
|l108_morceau                |      27|     0|        |          |         |
|l110-01_images_reflets      |      95|     0|        |          |         |
|l110-02_images_hommage      |      76|     0|        |          |         |
|l110-03_images_mouvement    |     177|     0|        |          |         |
|l111-01_images_cloches      |      49|     0|        |          |         |
|l111-02_images_lune         |      57|     0|        |          |         |
|l111-03_images_poissons     |      97|     0|        |          |         |
|l113-01_childrens_doctor    |      76|     0|        |          |         |
|l113-02_childrens_jimbos    |      81|     0|        |          |         |
|l113-03_childrens_serenade  |     124|     0|        |          |         |
|l113-04_childrens_snow      |      74|     0|        |          |         |
|l113-05_childrens_little    |      31|     0|        |          |         |
|l113-06_childrens_golliwoggs|     128|     0|        |          |         |
|l114_petit                  |      87|     0|        |          |         |
|l115_hommage                |     118|     0|        |          |         |
|l117-01_preludes_danseuses  |      31|     0|        |          |         |
|l117-02_preludes_voiles     |      64|     0|        |          |         |
|l117-03_preludes_vent       |      59|     0|        |          |         |
|l117-04_preludes_sons       |      53|     0|        |          |         |
|l117-05_preludes_collines   |      96|     0|        |          |         |
|l117-06_preludes_pas        |      36|     0|        |          |         |
|l117-07_preludes_ce         |      70|     0|        |          |         |
|l117-08_preludes_fille      |      39|     0|        |          |         |
|l117-09_preludes_serenade   |     137|     0|        |          |         |
|l117-10_preludes_cathedrale |      89|     0|        |          |         |
|l117-11_preludes_danse      |      96|     0|        |          |         |
|l117-12_preludes_minstrels  |      89|     0|        |          |         |
|l121_plus                   |     148|     0|        |          |         |
|l123-01_preludes_brouillards|      52|     0|        |          |         |
|l123-02_preludes_feuilles   |      52|     0|        |          |         |
|l123-03_preludes_puerta     |      90|     0|        |          |         |
|l123-04_preludes_fees       |     127|     0|        |          |         |
|l123-05_preludes_bruyeres   |      51|     0|        |          |         |
|l123-06_preludes_general    |     109|     0|        |          |         |
|l123-07_preludes_terrasse   |      45|     0|        |          |         |
|l123-08_preludes_ondine     |      74|     0|        |          |         |
|l123-09_preludes_hommage    |      54|     0|        |          |         |
|l123-10_preludes_canope     |      33|     0|        |          |         |
|l123-11_preludes_tierces    |     165|     0|        |          |         |
|l123-12_preludes_feux       |     100|     0|        |          |         |
|l132_berceuse               |      68|     0|        |          |         |
|l133_page                   |      38|     0|        |          |         |
|l136-01_etudes_cinq         |     116|     0|        |          |         |
|l136-02_etudes_tierces      |      76|     0|        |          |         |
|l136-03_etudes_quartes      |      85|     0|        |          |         |
|l136-04_etudes_sixtes       |      59|     0|        |          |         |
|l136-05_etudes_octaves      |     121|     0|        |          |         |
|l136-06_etudes_huit         |      68|     0|        |          |         |
|l136-07_etudes_degres       |      88|     0|        |          |         |
|l136-08_etudes_agrements    |      53|     0|        |          |         |
|l136-09_etudes_notes        |      84|     0|        |          |         |
|l136-10_etudes_sonorites    |      75|     0|        |          |         |
|l136-11_etudes_arpeges      |      67|     0|        |          |         |
|l136-12_etudes_accords      |     181|     0|        |          |         |
|l138_elegie                 |      21|     0|        |          |         |
