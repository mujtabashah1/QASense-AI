# NASA dataset audit

Raw files are preserved. Invalid numeric tokens and infinities become missing values; median imputation is fit inside training folds. Outliers, negative values, duplicate rows and conflicting labels are retained and documented, not silently deleted. Near constant means at least 99% one value; high correlation means |r| >= 0.95; extreme means outside 3 IQR.

## cm1

```json
{
  "filename": "cm1.csv",
  "sha256": "c05e4834217c7b0af59f4615c4972b36ef636d2f8414141de6e523bb82e70ac5",
  "rows": 498,
  "columns": 22,
  "column_names": [
    "loc",
    "v(g)",
    "ev(g)",
    "iv(g)",
    "n",
    "v",
    "l",
    "d",
    "i",
    "e",
    "b",
    "t",
    "lOCode",
    "lOComment",
    "lOBlank",
    "locCodeAndComment",
    "uniq_Op",
    "uniq_Opnd",
    "total_Op",
    "total_Opnd",
    "branchCount",
    "defects"
  ],
  "dtypes": {
    "loc": "float64",
    "v(g)": "float64",
    "ev(g)": "float64",
    "iv(g)": "float64",
    "n": "float64",
    "v": "float64",
    "l": "float64",
    "d": "float64",
    "i": "float64",
    "e": "float64",
    "b": "float64",
    "t": "float64",
    "lOCode": "int64",
    "lOComment": "int64",
    "lOBlank": "int64",
    "locCodeAndComment": "int64",
    "uniq_Op": "float64",
    "uniq_Opnd": "float64",
    "total_Op": "float64",
    "total_Opnd": "float64",
    "branchCount": "float64",
    "defects": "bool"
  },
  "missing_values": {
    "loc": 0,
    "v(g)": 0,
    "ev(g)": 0,
    "iv(g)": 0,
    "n": 0,
    "v": 0,
    "l": 0,
    "d": 0,
    "i": 0,
    "e": 0,
    "b": 0,
    "t": 0,
    "lOCode": 0,
    "lOComment": 0,
    "lOBlank": 0,
    "locCodeAndComment": 0,
    "uniq_Op": 0,
    "uniq_Opnd": 0,
    "total_Op": 0,
    "total_Opnd": 0,
    "branchCount": 0,
    "defects": 0
  },
  "duplicate_rows": 56,
  "target_distribution": {
    "0": 449,
    "1": 49
  },
  "defect_percentage": 9.839357429718875,
  "numeric_statistics": {
    "loc": {
      "count": 498.0,
      "mean": 29.644779116465863,
      "std": 42.75357237184671,
      "min": 1.0,
      "25%": 8.0,
      "50%": 17.0,
      "75%": 31.0,
      "max": 423.0
    },
    "v(g)": {
      "count": 498.0,
      "mean": 5.382329317269076,
      "std": 8.34735882733802,
      "min": 1.0,
      "25%": 1.0,
      "50%": 3.0,
      "75%": 6.0,
      "max": 96.0
    },
    "ev(g)": {
      "count": 498.0,
      "mean": 2.4907630522088353,
      "std": 3.6588465351648254,
      "min": 1.0,
      "25%": 1.0,
      "50%": 1.0,
      "75%": 1.0,
      "max": 30.0
    },
    "iv(g)": {
      "count": 498.0,
      "mean": 3.5289156626506024,
      "std": 5.464397963134891,
      "min": 1.0,
      "25%": 1.0,
      "50%": 2.0,
      "75%": 4.0,
      "max": 63.0
    },
    "n": {
      "count": 498.0,
      "mean": 143.95642570281126,
      "std": 221.04988804999883,
      "min": 1.0,
      "25%": 25.0,
      "50%": 67.5,
      "75%": 151.75,
      "max": 2075.0
    },
    "v": {
      "count": 498.0,
      "mean": 900.1758232931728,
      "std": 1690.8143335562245,
      "min": 0.0,
      "25%": 102.19,
      "50%": 329.82,
      "75%": 861.46,
      "max": 17124.28
    },
    "l": {
      "count": 498.0,
      "mean": 0.1463253012048193,
      "std": 0.15933700224403097,
      "min": 0.0,
      "25%": 0.05,
      "50%": 0.09,
      "75%": 0.1775,
      "max": 1.3
    },
    "d": {
      "count": 498.0,
      "mean": 15.829377510040159,
      "std": 15.330960462615966,
      "min": 0.0,
      "25%": 5.63,
      "50%": 11.64,
      "75%": 21.142500000000002,
      "max": 125.77
    },
    "i": {
      "count": 498.0,
      "mean": 38.45536144578313,
      "std": 36.99629707896739,
      "min": 0.0,
      "25%": 16.21,
      "50%": 27.4,
      "75%": 46.9,
      "max": 293.68
    },
    "e": {
      "count": 498.0,
      "mean": 34884.932329317264,
      "std": 134164.66559196552,
      "min": 0.0,
      "25%": 606.1700000000001,
      "50%": 3677.62,
      "75%": 16633.3375,
      "max": 2153690.63
    },
    "b": {
      "count": 498.0,
      "mean": 0.3046987951807229,
      "std": 0.5659983519810156,
      "min": 0.0,
      "25%": 0.03,
      "50%": 0.11,
      "75%": 0.3,
      "max": 5.71
    },
    "t": {
      "count": 498.0,
      "mean": 1938.056124497992,
      "std": 7453.591519015655,
      "min": 0.0,
      "25%": 33.6725,
      "50%": 204.31,
      "75%": 924.075,
      "max": 119649.48
    },
    "lOCode": {
      "count": 498.0,
      "mean": 3.78714859437751,
      "std": 8.508657888077655,
      "min": 0.0,
      "25%": 0.0,
      "50%": 1.0,
      "75%": 4.0,
      "max": 80.0
    },
    "lOComment": {
      "count": 498.0,
      "mean": 12.283132530120483,
      "std": 25.828605416730205,
      "min": 0.0,
      "25%": 0.0,
      "50%": 4.0,
      "75%": 14.0,
      "max": 339.0
    },
    "lOBlank": {
      "count": 498.0,
      "mean": 11.534136546184738,
      "std": 19.98147631512298,
      "min": 0.0,
      "25%": 1.0,
      "50%": 5.0,
      "75%": 13.0,
      "max": 164.0
    },
    "locCodeAndComment": {
      "count": 498.0,
      "mean": 0.006024096385542169,
      "std": 0.10011992518073475,
      "min": 0.0,
      "25%": 0.0,
      "50%": 0.0,
      "75%": 0.0,
      "max": 2.0
    },
    "uniq_Op": {
      "count": 498.0,
      "mean": 15.199196787148594,
      "std": 9.617815427581455,
      "min": 1.0,
      "25%": 9.0,
      "50%": 14.0,
      "75%": 20.0,
      "max": 72.0
    },
    "uniq_Opnd": {
      "count": 498.0,
      "mean": 25.452208835341366,
      "std": 33.92581570082524,
      "min": 0.0,
      "25%": 7.0,
      "50%": 15.0,
      "75%": 30.0,
      "max": 314.0
    },
    "total_Op": {
      "count": 498.0,
      "mean": 88.38995983935743,
      "std": 134.91751343762434,
      "min": 1.0,
      "25%": 15.0,
      "50%": 42.0,
      "75%": 94.75,
      "max": 1261.0
    },
    "total_Opnd": {
      "count": 498.0,
      "mean": 55.57068273092369,
      "std": 86.96952739961519,
      "min": 0.0,
      "25%": 10.0,
      "50%": 26.0,
      "75%": 59.75,
      "max": 814.0
    },
    "branchCount": {
      "count": 498.0,
      "mean": 9.348192771084337,
      "std": 15.072219396365654,
      "min": 1.0,
      "25%": 1.0,
      "50%": 5.0,
      "75%": 11.0,
      "max": 162.0
    }
  },
  "constant_or_near_constant": [
    "locCodeAndComment"
  ],
  "high_correlations": [
    {
      "a": "loc",
      "b": "v",
      "correlation": 0.9526985558108229
    },
    {
      "a": "v(g)",
      "b": "branchCount",
      "correlation": 0.991826086478643
    },
    {
      "a": "n",
      "b": "v",
      "correlation": 0.9933056162263377
    },
    {
      "a": "n",
      "b": "b",
      "correlation": 0.9839275175036554
    },
    {
      "a": "n",
      "b": "total_Op",
      "correlation": 0.9975627032718417
    },
    {
      "a": "n",
      "b": "total_Opnd",
      "correlation": 0.9941238226068079
    },
    {
      "a": "v",
      "b": "b",
      "correlation": 0.9915508973064833
    },
    {
      "a": "v",
      "b": "total_Op",
      "correlation": 0.990824024199339
    },
    {
      "a": "v",
      "b": "total_Opnd",
      "correlation": 0.9875681469548592
    },
    {
      "a": "e",
      "b": "t",
      "correlation": 0.9999999999566418
    },
    {
      "a": "b",
      "b": "total_Op",
      "correlation": 0.9814534172069818
    },
    {
      "a": "b",
      "b": "total_Opnd",
      "correlation": 0.9783677029278063
    },
    {
      "a": "total_Op",
      "b": "total_Opnd",
      "correlation": 0.9841478895113495
    }
  ],
  "invalid_numeric_values": {
    "loc": 0,
    "v(g)": 0,
    "ev(g)": 0,
    "iv(g)": 0,
    "n": 0,
    "v": 0,
    "l": 0,
    "d": 0,
    "i": 0,
    "e": 0,
    "b": 0,
    "t": 0,
    "lOCode": 0,
    "lOComment": 0,
    "lOBlank": 0,
    "locCodeAndComment": 0,
    "uniq_Op": 0,
    "uniq_Opnd": 0,
    "total_Op": 0,
    "total_Opnd": 0,
    "branchCount": 0
  },
  "infinite_values": {
    "loc": 0,
    "v(g)": 0,
    "ev(g)": 0,
    "iv(g)": 0,
    "n": 0,
    "v": 0,
    "l": 0,
    "d": 0,
    "i": 0,
    "e": 0,
    "b": 0,
    "t": 0,
    "lOCode": 0,
    "lOComment": 0,
    "lOBlank": 0,
    "locCodeAndComment": 0,
    "uniq_Op": 0,
    "uniq_Opnd": 0,
    "total_Op": 0,
    "total_Opnd": 0,
    "branchCount": 0
  },
  "negative_values": {
    "loc": 0,
    "v(g)": 0,
    "ev(g)": 0,
    "iv(g)": 0,
    "n": 0,
    "v": 0,
    "l": 0,
    "d": 0,
    "i": 0,
    "e": 0,
    "b": 0,
    "t": 0,
    "lOCode": 0,
    "lOComment": 0,
    "lOBlank": 0,
    "locCodeAndComment": 0,
    "uniq_Op": 0,
    "uniq_Opnd": 0,
    "total_Op": 0,
    "total_Opnd": 0,
    "branchCount": 0
  },
  "extreme_outliers_3_iqr": {
    "loc": 26,
    "v(g)": 17,
    "ev(g)": 123,
    "iv(g)": 18,
    "n": 30,
    "v": 38,
    "l": 13,
    "d": 7,
    "i": 14,
    "e": 58,
    "b": 33,
    "t": 58,
    "lOCode": 25,
    "lOComment": 23,
    "lOBlank": 20,
    "locCodeAndComment": 2,
    "uniq_Op": 2,
    "uniq_Opnd": 21,
    "total_Op": 25,
    "total_Opnd": 30,
    "branchCount": 14
  }
}
```

## jm1

```json
{
  "filename": "jm1.csv",
  "sha256": "013cb035b07f176e217c0db3abeaeeb214d262e1a63b237a44737ea0f301cc03",
  "rows": 13204,
  "columns": 22,
  "column_names": [
    "loc",
    "v(g)",
    "ev(g)",
    "iv(g)",
    "n",
    "v",
    "l",
    "d",
    "i",
    "e",
    "b",
    "t",
    "lOCode",
    "lOComment",
    "lOBlank",
    "locCodeAndComment",
    "uniq_Op",
    "uniq_Opnd",
    "total_Op",
    "total_Opnd",
    "branchCount",
    "defects"
  ],
  "dtypes": {
    "loc": "float64",
    "v(g)": "float64",
    "ev(g)": "float64",
    "iv(g)": "float64",
    "n": "float64",
    "v": "float64",
    "l": "float64",
    "d": "float64",
    "i": "float64",
    "e": "float64",
    "b": "float64",
    "t": "float64",
    "lOCode": "int64",
    "lOComment": "int64",
    "lOBlank": "int64",
    "locCodeAndComment": "int64",
    "uniq_Op": "float64",
    "uniq_Opnd": "float64",
    "total_Op": "float64",
    "total_Opnd": "float64",
    "branchCount": "float64",
    "defects": "bool"
  },
  "missing_values": {
    "loc": 0,
    "v(g)": 0,
    "ev(g)": 0,
    "iv(g)": 0,
    "n": 0,
    "v": 0,
    "l": 0,
    "d": 0,
    "i": 0,
    "e": 0,
    "b": 0,
    "t": 0,
    "lOCode": 0,
    "lOComment": 0,
    "lOBlank": 0,
    "locCodeAndComment": 0,
    "uniq_Op": 0,
    "uniq_Opnd": 0,
    "total_Op": 0,
    "total_Opnd": 0,
    "branchCount": 0,
    "defects": 0
  },
  "duplicate_rows": 4296,
  "target_distribution": {
    "0": 11101,
    "1": 2103
  },
  "defect_percentage": 15.926991820660405,
  "numeric_statistics": {
    "loc": {
      "count": 13204.0,
      "mean": 39.67752953650409,
      "std": 71.89936391140897,
      "min": 1.0,
      "25%": 10.0,
      "50%": 22.0,
      "75%": 43.0,
      "max": 3442.0
    },
    "v(g)": {
      "count": 13204.0,
      "mean": 6.020251438957891,
      "std": 12.194974103197527,
      "min": 1.0,
      "25%": 1.0,
      "50%": 3.0,
      "75%": 6.0,
      "max": 470.0
    },
    "ev(g)": {
      "count": 13204.0,
      "mean": 3.213677673432293,
      "std": 6.466248342701634,
      "min": 1.0,
      "25%": 1.0,
      "50%": 1.0,
      "75%": 3.0,
      "max": 165.0
    },
    "iv(g)": {
      "count": 13204.0,
      "mean": 3.8175098455013634,
      "std": 8.478262708458665,
      "min": 1.0,
      "25%": 1.0,
      "50%": 2.0,
      "75%": 4.0,
      "max": 402.0
    },
    "n": {
      "count": 13204.0,
      "mean": 108.86483641320812,
      "std": 234.80911059261533,
      "min": 0.0,
      "25%": 14.0,
      "50%": 47.0,
      "75%": 115.0,
      "max": 8441.0
    },
    "v": {
      "count": 13204.0,
      "mean": 633.893629960618,
      "std": 1804.952017347224,
      "min": 0.0,
      "25%": 49.83,
      "50%": 206.57999999999998,
      "75%": 593.4875,
      "max": 80843.08
    },
    "l": {
      "count": 13204.0,
      "mean": 0.14437443199030597,
      "std": 0.16530411745190454,
      "min": 0.0,
      "25%": 0.04,
      "50%": 0.08,
      "75%": 0.18,
      "max": 1.3
    },
    "d": {
      "count": 13204.0,
      "mean": 13.858345955770979,
      "std": 18.44510198661148,
      "min": 0.0,
      "25%": 3.0,
      "50%": 8.895,
      "75%": 18.24,
      "max": 418.2
    },
    "i": {
      "count": 13204.0,
      "mean": 28.757208421690393,
      "std": 32.50239585958356,
      "min": 0.0,
      "25%": 12.27,
      "50%": 21.515,
      "75%": 35.97,
      "max": 569.78
    },
    "e": {
      "count": 13204.0,
      "mean": 33656.99124734929,
      "std": 401839.4777303416,
      "min": 0.0,
      "25%": 159.45,
      "50%": 1877.33,
      "75%": 10492.595,
      "max": 31079782.27
    },
    "b": {
      "count": 13204.0,
      "mean": 0.21144956073916993,
      "std": 0.6017565023218324,
      "min": 0.0,
      "25%": 0.02,
      "50%": 0.07,
      "75%": 0.2,
      "max": 26.95
    },
    "t": {
      "count": 13204.0,
      "mean": 1869.832930172675,
      "std": 22324.41536300706,
      "min": 0.0,
      "25%": 8.86,
      "50%": 104.295,
      "75%": 582.9200000000001,
      "max": 1726654.57
    },
    "lOCode": {
      "count": 13204.0,
      "mean": 25.09921235989094,
      "std": 55.68910331033843,
      "min": 0.0,
      "25%": 3.0,
      "50%": 12.0,
      "75%": 27.0,
      "max": 2824.0
    },
    "lOComment": {
      "count": 13204.0,
      "mean": 2.59588003635262,
      "std": 8.693226635798707,
      "min": 0.0,
      "25%": 0.0,
      "50%": 0.0,
      "75%": 1.0,
      "max": 344.0
    },
    "lOBlank": {
      "count": 13204.0,
      "mean": 4.432444713723114,
      "std": 9.701586789634618,
      "min": 0.0,
      "25%": 0.0,
      "50%": 2.0,
      "75%": 5.0,
      "max": 447.0
    },
    "locCodeAndComment": {
      "count": 13204.0,
      "mean": 0.3522417449257801,
      "std": 1.8086683072212224,
      "min": 0.0,
      "25%": 0.0,
      "50%": 0.0,
      "75%": 0.0,
      "max": 108.0
    },
    "uniq_Op": {
      "count": 13204.0,
      "mean": 11.022887003938202,
      "std": 9.537748835424502,
      "min": 0.0,
      "25%": 5.0,
      "50%": 10.0,
      "75%": 15.0,
      "max": 411.0
    },
    "uniq_Opnd": {
      "count": 13204.0,
      "mean": 16.154286579824298,
      "std": 24.91815387910041,
      "min": 0.0,
      "25%": 4.0,
      "50%": 11.0,
      "75%": 20.0,
      "max": 1026.0
    },
    "total_Op": {
      "count": 13204.0,
      "mean": 64.77765828536806,
      "std": 142.72748781831967,
      "min": 0.0,
      "25%": 8.0,
      "50%": 28.0,
      "75%": 69.0,
      "max": 5420.0
    },
    "total_Opnd": {
      "count": 13204.0,
      "mean": 44.20298394425931,
      "std": 94.31132235678945,
      "min": 0.0,
      "25%": 6.0,
      "50%": 19.0,
      "75%": 46.0,
      "max": 3021.0
    },
    "branchCount": {
      "count": 13204.0,
      "mean": 10.669675855801271,
      "std": 21.289988448457972,
      "min": 1.0,
      "25%": 1.0,
      "50%": 5.0,
      "75%": 11.0,
      "max": 826.0
    }
  },
  "constant_or_near_constant": [],
  "high_correlations": [
    {
      "a": "v(g)",
      "b": "branchCount",
      "correlation": 0.9726411048655278
    },
    {
      "a": "n",
      "b": "v",
      "correlation": 0.9839553605124837
    },
    {
      "a": "n",
      "b": "b",
      "correlation": 0.9836323031566631
    },
    {
      "a": "n",
      "b": "total_Op",
      "correlation": 0.9955576868117327
    },
    {
      "a": "n",
      "b": "total_Opnd",
      "correlation": 0.9891445439398499
    },
    {
      "a": "v",
      "b": "b",
      "correlation": 0.9997084497794176
    },
    {
      "a": "v",
      "b": "lOCode",
      "correlation": 0.9606623145243024
    },
    {
      "a": "v",
      "b": "total_Op",
      "correlation": 0.9808512655563825
    },
    {
      "a": "v",
      "b": "total_Opnd",
      "correlation": 0.9716797649527878
    },
    {
      "a": "e",
      "b": "t",
      "correlation": 0.9999999999998058
    },
    {
      "a": "b",
      "b": "lOCode",
      "correlation": 0.96038196027545
    },
    {
      "a": "b",
      "b": "total_Op",
      "correlation": 0.9805361132174929
    },
    {
      "a": "b",
      "b": "total_Opnd",
      "correlation": 0.9713573883222218
    },
    {
      "a": "total_Op",
      "b": "total_Opnd",
      "correlation": 0.9718061372769123
    }
  ],
  "invalid_numeric_values": {
    "loc": 0,
    "v(g)": 0,
    "ev(g)": 0,
    "iv(g)": 0,
    "n": 0,
    "v": 0,
    "l": 0,
    "d": 0,
    "i": 0,
    "e": 0,
    "b": 0,
    "t": 0,
    "lOCode": 0,
    "lOComment": 0,
    "lOBlank": 0,
    "locCodeAndComment": 0,
    "uniq_Op": 0,
    "uniq_Opnd": 0,
    "total_Op": 0,
    "total_Opnd": 0,
    "branchCount": 0
  },
  "infinite_values": {
    "loc": 0,
    "v(g)": 0,
    "ev(g)": 0,
    "iv(g)": 0,
    "n": 0,
    "v": 0,
    "l": 0,
    "d": 0,
    "i": 0,
    "e": 0,
    "b": 0,
    "t": 0,
    "lOCode": 0,
    "lOComment": 0,
    "lOBlank": 0,
    "locCodeAndComment": 0,
    "uniq_Op": 0,
    "uniq_Opnd": 0,
    "total_Op": 0,
    "total_Opnd": 0,
    "branchCount": 0
  },
  "negative_values": {
    "loc": 0,
    "v(g)": 0,
    "ev(g)": 0,
    "iv(g)": 0,
    "n": 0,
    "v": 0,
    "l": 0,
    "d": 0,
    "i": 0,
    "e": 0,
    "b": 0,
    "t": 0,
    "lOCode": 0,
    "lOComment": 0,
    "lOBlank": 0,
    "locCodeAndComment": 0,
    "uniq_Op": 0,
    "uniq_Opnd": 0,
    "total_Op": 0,
    "total_Opnd": 0,
    "branchCount": 0
  },
  "extreme_outliers_3_iqr": {
    "loc": 572,
    "v(g)": 560,
    "ev(g)": 936,
    "iv(g)": 495,
    "n": 628,
    "v": 767,
    "l": 527,
    "d": 242,
    "i": 305,
    "e": 1365,
    "b": 762,
    "t": 1365,
    "lOCode": 556,
    "lOComment": 1965,
    "lOBlank": 474,
    "locCodeAndComment": 1556,
    "uniq_Op": 32,
    "uniq_Opnd": 342,
    "total_Op": 614,
    "total_Opnd": 654,
    "branchCount": 536
  }
}
```

## kc1

```json
{
  "filename": "kc1.csv",
  "sha256": "bdc2221d5490243bd033baaf35a1a906a460cd88a54444dfe8be92aba430e798",
  "rows": 2109,
  "columns": 22,
  "column_names": [
    "loc",
    "v(g)",
    "ev(g)",
    "iv(g)",
    "n",
    "v",
    "l",
    "d",
    "i",
    "e",
    "b",
    "t",
    "lOCode",
    "lOComment",
    "lOBlank",
    "locCodeAndComment",
    "uniq_Op",
    "uniq_Opnd",
    "total_Op",
    "total_Opnd",
    "branchCount",
    "defects"
  ],
  "dtypes": {
    "loc": "float64",
    "v(g)": "float64",
    "ev(g)": "float64",
    "iv(g)": "float64",
    "n": "float64",
    "v": "float64",
    "l": "float64",
    "d": "float64",
    "i": "float64",
    "e": "float64",
    "b": "float64",
    "t": "float64",
    "lOCode": "int64",
    "lOComment": "int64",
    "lOBlank": "int64",
    "locCodeAndComment": "int64",
    "uniq_Op": "float64",
    "uniq_Opnd": "float64",
    "total_Op": "float64",
    "total_Opnd": "float64",
    "branchCount": "float64",
    "defects": "bool"
  },
  "missing_values": {
    "loc": 0,
    "v(g)": 0,
    "ev(g)": 0,
    "iv(g)": 0,
    "n": 0,
    "v": 0,
    "l": 0,
    "d": 0,
    "i": 0,
    "e": 0,
    "b": 0,
    "t": 0,
    "lOCode": 0,
    "lOComment": 0,
    "lOBlank": 0,
    "locCodeAndComment": 0,
    "uniq_Op": 0,
    "uniq_Opnd": 0,
    "total_Op": 0,
    "total_Opnd": 0,
    "branchCount": 0,
    "defects": 0
  },
  "duplicate_rows": 897,
  "target_distribution": {
    "0": 1783,
    "1": 326
  },
  "defect_percentage": 15.457562825983878,
  "numeric_statistics": {
    "loc": {
      "count": 2109.0,
      "mean": 20.37226173541963,
      "std": 29.754442107025678,
      "min": 1.0,
      "25%": 3.0,
      "50%": 9.0,
      "75%": 24.0,
      "max": 288.0
    },
    "v(g)": {
      "count": 2109.0,
      "mean": 2.8380275011853957,
      "std": 3.900763207714471,
      "min": 1.0,
      "25%": 1.0,
      "50%": 1.0,
      "75%": 3.0,
      "max": 45.0
    },
    "ev(g)": {
      "count": 2109.0,
      "mean": 1.674442863916548,
      "std": 2.2006589135696126,
      "min": 1.0,
      "25%": 1.0,
      "50%": 1.0,
      "75%": 1.0,
      "max": 26.0
    },
    "iv(g)": {
      "count": 2109.0,
      "mean": 2.546420104314841,
      "std": 3.375859260340955,
      "min": 1.0,
      "25%": 1.0,
      "50%": 1.0,
      "75%": 3.0,
      "max": 45.0
    },
    "n": {
      "count": 2109.0,
      "mean": 49.829445234708395,
      "std": 83.59987420165592,
      "min": 0.0,
      "25%": 4.0,
      "50%": 16.0,
      "75%": 58.0,
      "max": 1106.0
    },
    "v": {
      "count": 2109.0,
      "mean": 258.69671882408727,
      "std": 516.3176046089964,
      "min": 0.0,
      "25%": 8.0,
      "50%": 57.06,
      "75%": 265.93,
      "max": 7918.82
    },
    "l": {
      "count": 2109.0,
      "mean": 0.31958274063537223,
      "std": 0.3170290459009352,
      "min": 0.0,
      "25%": 0.08,
      "50%": 0.2,
      "75%": 0.67,
      "max": 2.0
    },
    "d": {
      "count": 2109.0,
      "mean": 6.771242294926505,
      "std": 7.863645549059695,
      "min": 0.0,
      "25%": 1.5,
      "50%": 3.5,
      "75%": 9.2,
      "max": 53.75
    },
    "i": {
      "count": 2109.0,
      "mean": 21.24007112375533,
      "std": 21.500366912377462,
      "min": 0.0,
      "25%": 5.33,
      "50%": 14.4,
      "75%": 29.85,
      "max": 193.06
    },
    "e": {
      "count": 2109.0,
      "mean": 5242.386239924134,
      "std": 17444.981211378636,
      "min": 0.0,
      "25%": 12.0,
      "50%": 213.97,
      "75%": 2276.02,
      "max": 324803.51
    },
    "b": {
      "count": 2109.0,
      "mean": 0.08673779042200096,
      "std": 0.1755065263631647,
      "min": 0.0,
      "25%": 0.0,
      "50%": 0.02,
      "75%": 0.09,
      "max": 2.64
    },
    "t": {
      "count": 2109.0,
      "mean": 291.2450403034614,
      "std": 969.1651602540443,
      "min": 0.0,
      "25%": 0.67,
      "50%": 11.89,
      "75%": 126.45,
      "max": 18044.64
    },
    "lOCode": {
      "count": 2109.0,
      "mean": 14.525367472735894,
      "std": 24.18830198588257,
      "min": 0.0,
      "25%": 0.0,
      "50%": 5.0,
      "75%": 17.0,
      "max": 262.0
    },
    "lOComment": {
      "count": 2109.0,
      "mean": 0.9459459459459459,
      "std": 3.0852712044561676,
      "min": 0.0,
      "25%": 0.0,
      "50%": 0.0,
      "75%": 0.0,
      "max": 44.0
    },
    "lOBlank": {
      "count": 2109.0,
      "mean": 1.759601706970128,
      "std": 3.856850165604205,
      "min": 0.0,
      "25%": 0.0,
      "50%": 0.0,
      "75%": 2.0,
      "max": 58.0
    },
    "locCodeAndComment": {
      "count": 2109.0,
      "mean": 0.1327643432906591,
      "std": 0.7040227268649906,
      "min": 0.0,
      "25%": 0.0,
      "50%": 0.0,
      "75%": 0.0,
      "max": 12.0
    },
    "uniq_Op": {
      "count": 2109.0,
      "mean": 7.631673779042201,
      "std": 5.73034650423644,
      "min": 0.0,
      "25%": 3.0,
      "50%": 6.0,
      "75%": 11.0,
      "max": 37.0
    },
    "uniq_Opnd": {
      "count": 2109.0,
      "mean": 9.537316263632054,
      "std": 12.195726559109909,
      "min": 0.0,
      "25%": 1.0,
      "50%": 5.0,
      "75%": 13.0,
      "max": 120.0
    },
    "total_Op": {
      "count": 2109.0,
      "mean": 31.043717401612138,
      "std": 51.776055918156075,
      "min": 0.0,
      "25%": 3.0,
      "50%": 10.0,
      "75%": 36.0,
      "max": 678.0
    },
    "total_Opnd": {
      "count": 2109.0,
      "mean": 18.786723565670933,
      "std": 32.07439842427073,
      "min": 0.0,
      "25%": 1.0,
      "50%": 6.0,
      "75%": 22.0,
      "max": 428.0
    },
    "branchCount": {
      "count": 2109.0,
      "mean": 4.665908013276434,
      "std": 7.7922064524502,
      "min": 1.0,
      "25%": 1.0,
      "50%": 1.0,
      "75%": 5.0,
      "max": 89.0
    }
  },
  "constant_or_near_constant": [],
  "high_correlations": [
    {
      "a": "loc",
      "b": "lOCode",
      "correlation": 0.9854910962489218
    },
    {
      "a": "v(g)",
      "b": "iv(g)",
      "correlation": 0.9656894444920697
    },
    {
      "a": "v(g)",
      "b": "branchCount",
      "correlation": 0.9986080128863948
    },
    {
      "a": "iv(g)",
      "b": "branchCount",
      "correlation": 0.9647333260566909
    },
    {
      "a": "n",
      "b": "v",
      "correlation": 0.9949563978400988
    },
    {
      "a": "n",
      "b": "b",
      "correlation": 0.9737113509539094
    },
    {
      "a": "n",
      "b": "lOCode",
      "correlation": 0.9610503083214567
    },
    {
      "a": "n",
      "b": "total_Op",
      "correlation": 0.9981462393504876
    },
    {
      "a": "n",
      "b": "total_Opnd",
      "correlation": 0.9951618820839006
    },
    {
      "a": "v",
      "b": "b",
      "correlation": 0.9789578561559846
    },
    {
      "a": "v",
      "b": "total_Op",
      "correlation": 0.9920459440136783
    },
    {
      "a": "v",
      "b": "total_Opnd",
      "correlation": 0.9918660389637683
    },
    {
      "a": "e",
      "b": "t",
      "correlation": 0.9999999993905164
    },
    {
      "a": "b",
      "b": "total_Op",
      "correlation": 0.9708872457655388
    },
    {
      "a": "b",
      "b": "total_Opnd",
      "correlation": 0.9708525008339055
    },
    {
      "a": "lOCode",
      "b": "total_Op",
      "correlation": 0.9611871834361685
    },
    {
      "a": "lOCode",
      "b": "total_Opnd",
      "correlation": 0.9533044904189532
    },
    {
      "a": "uniq_Opnd",
      "b": "total_Opnd",
      "correlation": 0.9524682590790727
    },
    {
      "a": "total_Op",
      "b": "total_Opnd",
      "correlation": 0.987337865626014
    }
  ],
  "invalid_numeric_values": {
    "loc": 0,
    "v(g)": 0,
    "ev(g)": 0,
    "iv(g)": 0,
    "n": 0,
    "v": 0,
    "l": 0,
    "d": 0,
    "i": 0,
    "e": 0,
    "b": 0,
    "t": 0,
    "lOCode": 0,
    "lOComment": 0,
    "lOBlank": 0,
    "locCodeAndComment": 0,
    "uniq_Op": 0,
    "uniq_Opnd": 0,
    "total_Op": 0,
    "total_Opnd": 0,
    "branchCount": 0
  },
  "infinite_values": {
    "loc": 0,
    "v(g)": 0,
    "ev(g)": 0,
    "iv(g)": 0,
    "n": 0,
    "v": 0,
    "l": 0,
    "d": 0,
    "i": 0,
    "e": 0,
    "b": 0,
    "t": 0,
    "lOCode": 0,
    "lOComment": 0,
    "lOBlank": 0,
    "locCodeAndComment": 0,
    "uniq_Op": 0,
    "uniq_Opnd": 0,
    "total_Op": 0,
    "total_Opnd": 0,
    "branchCount": 0
  },
  "negative_values": {
    "loc": 0,
    "v(g)": 0,
    "ev(g)": 0,
    "iv(g)": 0,
    "n": 0,
    "v": 0,
    "l": 0,
    "d": 0,
    "i": 0,
    "e": 0,
    "b": 0,
    "t": 0,
    "lOCode": 0,
    "lOComment": 0,
    "lOBlank": 0,
    "locCodeAndComment": 0,
    "uniq_Op": 0,
    "uniq_Opnd": 0,
    "total_Op": 0,
    "total_Opnd": 0,
    "branchCount": 0
  },
  "extreme_outliers_3_iqr": {
    "loc": 82,
    "v(g)": 125,
    "ev(g)": 284,
    "iv(g)": 106,
    "n": 94,
    "v": 132,
    "l": 0,
    "d": 37,
    "i": 16,
    "e": 267,
    "b": 119,
    "t": 267,
    "lOCode": 88,
    "lOComment": 491,
    "lOBlank": 109,
    "locCodeAndComment": 127,
    "uniq_Op": 1,
    "uniq_Opnd": 36,
    "total_Op": 95,
    "total_Opnd": 99,
    "branchCount": 124
  }
}
```

## kc2

```json
{
  "filename": "kc2.csv",
  "sha256": "fedb0f8b4e131dad110e2a7713f7d3b1c0efb68a55b7df7c469610998dafbec7",
  "rows": 522,
  "columns": 22,
  "column_names": [
    "loc",
    "v(g)",
    "ev(g)",
    "iv(g)",
    "n",
    "v",
    "l",
    "d",
    "i",
    "e",
    "b",
    "t",
    "lOCode",
    "lOComment",
    "lOBlank",
    "lOCodeAndComment",
    "uniq_Op",
    "uniq_Opnd",
    "total_Op",
    "total_Opnd",
    "branchCount",
    "problems"
  ],
  "dtypes": {
    "loc": "float64",
    "v(g)": "float64",
    "ev(g)": "float64",
    "iv(g)": "float64",
    "n": "float64",
    "v": "float64",
    "l": "float64",
    "d": "float64",
    "i": "float64",
    "e": "float64",
    "b": "float64",
    "t": "float64",
    "lOCode": "int64",
    "lOComment": "int64",
    "lOBlank": "int64",
    "lOCodeAndComment": "int64",
    "uniq_Op": "float64",
    "uniq_Opnd": "float64",
    "total_Op": "float64",
    "total_Opnd": "float64",
    "branchCount": "float64",
    "problems": "str"
  },
  "missing_values": {
    "loc": 0,
    "v(g)": 0,
    "ev(g)": 0,
    "iv(g)": 0,
    "n": 0,
    "v": 0,
    "l": 0,
    "d": 0,
    "i": 0,
    "e": 0,
    "b": 0,
    "t": 0,
    "lOCode": 0,
    "lOComment": 0,
    "lOBlank": 0,
    "lOCodeAndComment": 0,
    "uniq_Op": 0,
    "uniq_Opnd": 0,
    "total_Op": 0,
    "total_Opnd": 0,
    "branchCount": 0,
    "problems": 0
  },
  "duplicate_rows": 147,
  "target_distribution": {
    "0": 415,
    "1": 107
  },
  "defect_percentage": 20.49808429118774,
  "numeric_statistics": {
    "loc": {
      "count": 522.0,
      "mean": 36.894827586206894,
      "std": 77.9440475041743,
      "min": 1.0,
      "25%": 4.0,
      "50%": 13.0,
      "75%": 45.0,
      "max": 1275.0
    },
    "v(g)": {
      "count": 522.0,
      "mean": 4.893486590038314,
      "std": 10.97677165815744,
      "min": 1.0,
      "25%": 1.0,
      "50%": 2.0,
      "75%": 5.0,
      "max": 180.0
    },
    "ev(g)": {
      "count": 522.0,
      "mean": 2.4471264367816095,
      "std": 6.665003090693527,
      "min": 1.0,
      "25%": 1.0,
      "50%": 1.0,
      "75%": 1.0,
      "max": 125.0
    },
    "iv(g)": {
      "count": 522.0,
      "mean": 3.6501915708812263,
      "std": 8.054860477654332,
      "min": 1.0,
      "25%": 1.0,
      "50%": 2.0,
      "75%": 4.0,
      "max": 143.0
    },
    "n": {
      "count": 522.0,
      "mean": 94.6308429118774,
      "std": 233.23016546612754,
      "min": 1.0,
      "25%": 5.0,
      "50%": 27.0,
      "75%": 104.75,
      "max": 3982.0
    },
    "v": {
      "count": 522.0,
      "mean": 555.4726436781609,
      "std": 1817.4683201696398,
      "min": 0.0,
      "25%": 11.61,
      "50%": 109.205,
      "75%": 543.7875,
      "max": 33814.56
    },
    "l": {
      "count": 522.0,
      "mean": 0.274463601532567,
      "std": 0.2756089046934317,
      "min": 0.0,
      "25%": 0.06,
      "50%": 0.14,
      "75%": 0.5,
      "max": 2.0
    },
    "d": {
      "count": 522.0,
      "mean": 9.73528735632184,
      "std": 11.088639965731987,
      "min": 0.0,
      "25%": 1.5,
      "50%": 6.0,
      "75%": 14.61,
      "max": 103.53
    },
    "i": {
      "count": 522.0,
      "mean": 28.32044061302682,
      "std": 32.22559667995898,
      "min": 0.0,
      "25%": 7.74,
      "50%": 20.560000000000002,
      "75%": 38.364999999999995,
      "max": 415.06
    },
    "e": {
      "count": 522.0,
      "mean": 18542.99162835249,
      "std": 113271.21645142233,
      "min": 0.0,
      "25%": 17.41,
      "50%": 613.5899999999999,
      "75%": 7624.085,
      "max": 2147483.64
    },
    "b": {
      "count": 522.0,
      "mean": 0.1889655172413793,
      "std": 0.6087923311405048,
      "min": 0.0,
      "25%": 0.0,
      "50%": 0.04,
      "75%": 0.18,
      "max": 11.27
    },
    "t": {
      "count": 522.0,
      "mean": 1094.8114559386972,
      "std": 7556.52158121456,
      "min": 0.0,
      "25%": 0.97,
      "50%": 34.09,
      "75%": 423.5625,
      "max": 153047.01
    },
    "lOCode": {
      "count": 522.0,
      "mean": 27.772030651340994,
      "std": 64.4314850081845,
      "min": 0.0,
      "25%": 2.0,
      "50%": 8.0,
      "75%": 33.75,
      "max": 1107.0
    },
    "lOComment": {
      "count": 522.0,
      "mean": 2.0,
      "std": 5.582052402218492,
      "min": 0.0,
      "25%": 0.0,
      "50%": 0.0,
      "75%": 2.0,
      "max": 44.0
    },
    "lOBlank": {
      "count": 522.0,
      "mean": 4.339080459770115,
      "std": 9.21475312531488,
      "min": 0.0,
      "25%": 0.0,
      "50%": 1.0,
      "75%": 5.0,
      "max": 121.0
    },
    "locCodeAndComment": {
      "count": 522.0,
      "mean": 0.28160919540229884,
      "std": 1.0382358508120604,
      "min": 0.0,
      "25%": 0.0,
      "50%": 0.0,
      "75%": 0.0,
      "max": 11.0
    },
    "uniq_Op": {
      "count": 522.0,
      "mean": 9.197701149425287,
      "std": 6.360180413878306,
      "min": 1.0,
      "25%": 3.0,
      "50%": 8.0,
      "75%": 14.0,
      "max": 47.0
    },
    "uniq_Opnd": {
      "count": 522.0,
      "mean": 14.465900383141761,
      "std": 22.08666115145592,
      "min": 0.0,
      "25%": 2.0,
      "50%": 7.0,
      "75%": 20.0,
      "max": 325.0
    },
    "total_Op": {
      "count": 522.0,
      "mean": 57.61149425287356,
      "std": 142.99074135520084,
      "min": 1.0,
      "25%": 4.0,
      "50%": 16.5,
      "75%": 64.0,
      "max": 2469.0
    },
    "total_Opnd": {
      "count": 522.0,
      "mean": 37.02337164750958,
      "std": 90.39861984561563,
      "min": 0.0,
      "25%": 2.0,
      "50%": 11.0,
      "75%": 41.0,
      "max": 1513.0
    },
    "branchCount": {
      "count": 522.0,
      "mean": 8.765134099616857,
      "std": 21.942778727265367,
      "min": 1.0,
      "25%": 1.0,
      "50%": 3.0,
      "75%": 9.0,
      "max": 361.0
    }
  },
  "constant_or_near_constant": [],
  "high_correlations": [
    {
      "a": "loc",
      "b": "v(g)",
      "correlation": 0.9640481847521226
    },
    {
      "a": "loc",
      "b": "iv(g)",
      "correlation": 0.9635171935184073
    },
    {
      "a": "loc",
      "b": "n",
      "correlation": 0.9895285855167549
    },
    {
      "a": "loc",
      "b": "v",
      "correlation": 0.9753569499948566
    },
    {
      "a": "loc",
      "b": "b",
      "correlation": 0.9674985871793493
    },
    {
      "a": "loc",
      "b": "lOCode",
      "correlation": 0.992134287811029
    },
    {
      "a": "loc",
      "b": "uniq_Opnd",
      "correlation": 0.9648943061505875
    },
    {
      "a": "loc",
      "b": "total_Op",
      "correlation": 0.9929995226979642
    },
    {
      "a": "loc",
      "b": "total_Opnd",
      "correlation": 0.9923545353779212
    },
    {
      "a": "loc",
      "b": "branchCount",
      "correlation": 0.959667594021654
    },
    {
      "a": "v(g)",
      "b": "iv(g)",
      "correlation": 0.9765979195640209
    },
    {
      "a": "v(g)",
      "b": "n",
      "correlation": 0.9632785856038502
    },
    {
      "a": "v(g)",
      "b": "v",
      "correlation": 0.9585525692351852
    },
    {
      "a": "v(g)",
      "b": "b",
      "correlation": 0.9516102499378052
    },
    {
      "a": "v(g)",
      "b": "lOCode",
      "correlation": 0.9638154764949838
    },
    {
      "a": "v(g)",
      "b": "total_Op",
      "correlation": 0.9663263731187716
    },
    {
      "a": "v(g)",
      "b": "total_Opnd",
      "correlation": 0.9654391022321243
    },
    {
      "a": "v(g)",
      "b": "branchCount",
      "correlation": 0.9958863751847278
    },
    {
      "a": "ev(g)",
      "b": "iv(g)",
      "correlation": 0.9523054126496485
    },
    {
      "a": "iv(g)",
      "b": "n",
      "correlation": 0.9664575045736928
    },
    {
      "a": "iv(g)",
      "b": "v",
      "correlation": 0.9704481523193208
    },
    {
      "a": "iv(g)",
      "b": "b",
      "correlation": 0.9636839237005063
    },
    {
      "a": "iv(g)",
      "b": "lOCode",
      "correlation": 0.9682643851666578
    },
    {
      "a": "iv(g)",
      "b": "total_Op",
      "correlation": 0.9715973002410261
    },
    {
      "a": "iv(g)",
      "b": "total_Opnd",
      "correlation": 0.9643617828211886
    },
    {
      "a": "iv(g)",
      "b": "branchCount",
      "correlation": 0.9723210895657782
    },
    {
      "a": "n",
      "b": "v",
      "correlation": 0.9914311251465671
    },
    {
      "a": "n",
      "b": "b",
      "correlation": 0.9838978080878026
    },
    {
      "a": "n",
      "b": "lOCode",
      "correlation": 0.9949359989836996
    },
    {
      "a": "n",
      "b": "uniq_Opnd",
      "correlation": 0.9550481368783149
    },
    {
      "a": "n",
      "b": "total_Op",
      "correlation": 0.9963240367259358
    },
    {
      "a": "n",
      "b": "total_Opnd",
      "correlation": 0.9951453894284307
    },
    {
      "a": "n",
      "b": "branchCount",
      "correlation": 0.9666807420316575
    },
    {
      "a": "v",
      "b": "e",
      "correlation": 0.9691661592172411
    },
    {
      "a": "v",
      "b": "b",
      "correlation": 0.9930256316681441
    },
    {
      "a": "v",
      "b": "t",
      "correlation": 0.963933318177029
    },
    {
      "a": "v",
      "b": "lOCode",
      "correlation": 0.9859197856730803
    },
    {
      "a": "v",
      "b": "total_Op",
      "correlation": 0.9897101163879566
    },
    {
      "a": "v",
      "b": "total_Opnd",
      "correlation": 0.9849737947472699
    },
    {
      "a": "v",
      "b": "branchCount",
      "correlation": 0.9618511752285668
    },
    {
      "a": "i",
      "b": "uniq_Opnd",
      "correlation": 0.9559833612288439
    },
    {
      "a": "e",
      "b": "b",
      "correlation": 0.9632148114446291
    },
    {
      "a": "e",
      "b": "t",
      "correlation": 0.9938567651036144
    },
    {
      "a": "b",
      "b": "t",
      "correlation": 0.9581382256717957
    },
    {
      "a": "b",
      "b": "lOCode",
      "correlation": 0.9783404758275227
    },
    {
      "a": "b",
      "b": "total_Op",
      "correlation": 0.9822496885744729
    },
    {
      "a": "b",
      "b": "total_Opnd",
      "correlation": 0.9775287226810168
    },
    {
      "a": "b",
      "b": "branchCount",
      "correlation": 0.9547918384917594
    },
    {
      "a": "lOCode",
      "b": "uniq_Opnd",
      "correlation": 0.9558571026722393
    },
    {
      "a": "lOCode",
      "b": "total_Op",
      "correlation": 0.9913834758810031
    },
    {
      "a": "lOCode",
      "b": "total_Opnd",
      "correlation": 0.9890670165358532
    },
    {
      "a": "lOCode",
      "b": "branchCount",
      "correlation": 0.9668851236506705
    },
    {
      "a": "uniq_Opnd",
      "b": "total_Op",
      "correlation": 0.9562752205903545
    },
    {
      "a": "uniq_Opnd",
      "b": "total_Opnd",
      "correlation": 0.961808499986273
    },
    {
      "a": "total_Op",
      "b": "total_Opnd",
      "correlation": 0.9970976256436603
    },
    {
      "a": "total_Op",
      "b": "branchCount",
      "correlation": 0.9632525490236392
    },
    {
      "a": "total_Opnd",
      "b": "branchCount",
      "correlation": 0.9617118455310061
    }
  ],
  "invalid_numeric_values": {
    "loc": 0,
    "v(g)": 0,
    "ev(g)": 0,
    "iv(g)": 0,
    "n": 0,
    "v": 0,
    "l": 0,
    "d": 0,
    "i": 0,
    "e": 0,
    "b": 0,
    "t": 0,
    "lOCode": 0,
    "lOComment": 0,
    "lOBlank": 0,
    "locCodeAndComment": 0,
    "uniq_Op": 0,
    "uniq_Opnd": 0,
    "total_Op": 0,
    "total_Opnd": 0,
    "branchCount": 0
  },
  "infinite_values": {
    "loc": 0,
    "v(g)": 0,
    "ev(g)": 0,
    "iv(g)": 0,
    "n": 0,
    "v": 0,
    "l": 0,
    "d": 0,
    "i": 0,
    "e": 0,
    "b": 0,
    "t": 0,
    "lOCode": 0,
    "lOComment": 0,
    "lOBlank": 0,
    "locCodeAndComment": 0,
    "uniq_Op": 0,
    "uniq_Opnd": 0,
    "total_Op": 0,
    "total_Opnd": 0,
    "branchCount": 0
  },
  "negative_values": {
    "loc": 0,
    "v(g)": 0,
    "ev(g)": 0,
    "iv(g)": 0,
    "n": 0,
    "v": 0,
    "l": 0,
    "d": 0,
    "i": 0,
    "e": 0,
    "b": 0,
    "t": 0,
    "lOCode": 0,
    "lOComment": 0,
    "lOBlank": 0,
    "locCodeAndComment": 0,
    "uniq_Op": 0,
    "uniq_Opnd": 0,
    "total_Op": 0,
    "total_Opnd": 0,
    "branchCount": 0
  },
  "extreme_outliers_3_iqr": {
    "loc": 14,
    "v(g)": 25,
    "ev(g)": 98,
    "iv(g)": 17,
    "n": 21,
    "v": 26,
    "l": 1,
    "d": 4,
    "i": 4,
    "e": 52,
    "b": 27,
    "t": 52,
    "lOCode": 15,
    "lOComment": 29,
    "lOBlank": 15,
    "locCodeAndComment": 70,
    "uniq_Op": 0,
    "uniq_Opnd": 6,
    "total_Op": 21,
    "total_Opnd": 20,
    "branchCount": 25
  }
}
```

## pc1

```json
{
  "filename": "pc1.csv",
  "sha256": "9331ea04c83dfbcb618dbb707c8123e661f2ac5f59c85c03031f46fa0ce9f3e4",
  "rows": 1109,
  "columns": 22,
  "column_names": [
    "loc",
    "v(g)",
    "ev(g)",
    "iv(G)",
    "N",
    "V",
    "L",
    "D",
    "I",
    "E",
    "B",
    "T",
    "lOCode",
    "lOComment",
    "locCodeAndComment",
    "lOBlank",
    "uniq_Op",
    "uniq_Opnd",
    "total_Op",
    "total_Opnd",
    "branchCount",
    "defects"
  ],
  "dtypes": {
    "loc": "float64",
    "v(g)": "float64",
    "ev(g)": "float64",
    "iv(G)": "float64",
    "N": "float64",
    "V": "float64",
    "L": "float64",
    "D": "float64",
    "I": "float64",
    "E": "float64",
    "B": "float64",
    "T": "float64",
    "lOCode": "int64",
    "lOComment": "int64",
    "locCodeAndComment": "int64",
    "lOBlank": "int64",
    "uniq_Op": "float64",
    "uniq_Opnd": "float64",
    "total_Op": "float64",
    "total_Opnd": "float64",
    "branchCount": "float64",
    "defects": "bool"
  },
  "missing_values": {
    "loc": 0,
    "v(g)": 0,
    "ev(g)": 0,
    "iv(G)": 0,
    "N": 0,
    "V": 0,
    "L": 0,
    "D": 0,
    "I": 0,
    "E": 0,
    "B": 0,
    "T": 0,
    "lOCode": 0,
    "lOComment": 0,
    "locCodeAndComment": 0,
    "lOBlank": 0,
    "uniq_Op": 0,
    "uniq_Opnd": 0,
    "total_Op": 0,
    "total_Opnd": 0,
    "branchCount": 0,
    "defects": 0
  },
  "duplicate_rows": 155,
  "target_distribution": {
    "0": 1032,
    "1": 77
  },
  "defect_percentage": 6.943192064923355,
  "numeric_statistics": {
    "loc": {
      "count": 1109.0,
      "mean": 23.3761045987376,
      "std": 35.284017009012715,
      "min": 0.0,
      "25%": 7.0,
      "50%": 13.0,
      "75%": 26.0,
      "max": 602.0
    },
    "v(g)": {
      "count": 1109.0,
      "mean": 5.510730387736699,
      "std": 8.95897601184721,
      "min": 1.0,
      "25%": 1.0,
      "50%": 3.0,
      "75%": 6.0,
      "max": 136.0
    },
    "ev(g)": {
      "count": 1109.0,
      "mean": 2.7668169522091977,
      "std": 5.56770429392132,
      "min": 1.0,
      "25%": 1.0,
      "50%": 1.0,
      "75%": 3.0,
      "max": 123.0
    },
    "iv(g)": {
      "count": 1109.0,
      "mean": 3.321370604147881,
      "std": 6.40199818738095,
      "min": 1.0,
      "25%": 1.0,
      "50%": 2.0,
      "75%": 3.0,
      "max": 123.0
    },
    "n": {
      "count": 1109.0,
      "mean": 117.39341749323715,
      "std": 197.33688791069375,
      "min": 1.0,
      "25%": 25.0,
      "50%": 58.0,
      "75%": 126.0,
      "max": 2785.0
    },
    "v": {
      "count": 1109.0,
      "mean": 699.7112173128945,
      "std": 1509.5456837615839,
      "min": 0.0,
      "25%": 97.67,
      "50%": 275.1,
      "75%": 674.04,
      "max": 25942.69
    },
    "l": {
      "count": 1109.0,
      "mean": 0.12941388638412984,
      "std": 0.14695194033180747,
      "min": 0.0,
      "25%": 0.05,
      "50%": 0.08,
      "75%": 0.16,
      "max": 2.0
    },
    "d": {
      "count": 1109.0,
      "mean": 15.396266907123534,
      "std": 16.337508115698103,
      "min": 0.0,
      "25%": 5.71,
      "50%": 11.61,
      "75%": 20.5,
      "max": 270.66
    },
    "i": {
      "count": 1109.0,
      "mean": 32.904472497745715,
      "std": 35.387684803542655,
      "min": 0.0,
      "25%": 14.04,
      "50%": 23.57,
      "75%": 41.15,
      "max": 598.33
    },
    "e": {
      "count": 1109.0,
      "mean": 28822.882380522995,
      "std": 170643.60332778355,
      "min": 0.0,
      "25%": 558.13,
      "50%": 3189.17,
      "75%": 12381.56,
      "max": 4279633.01
    },
    "b": {
      "count": 1109.0,
      "mean": 0.23522993688007215,
      "std": 0.5047731255451794,
      "min": 0.0,
      "25%": 0.03,
      "50%": 0.09,
      "75%": 0.23,
      "max": 8.65
    },
    "t": {
      "count": 1109.0,
      "mean": 1601.2730297565374,
      "std": 9480.199988537977,
      "min": 0.0,
      "25%": 31.01,
      "50%": 177.18,
      "75%": 687.86,
      "max": 237757.4
    },
    "lOCode": {
      "count": 1109.0,
      "mean": 22.4346257889991,
      "std": 33.574644872463,
      "min": 0.0,
      "25%": 7.0,
      "50%": 13.0,
      "75%": 24.0,
      "max": 600.0
    },
    "lOComment": {
      "count": 1109.0,
      "mean": 4.696122633002705,
      "std": 10.518446634817154,
      "min": 0.0,
      "25%": 0.0,
      "50%": 0.0,
      "75%": 5.0,
      "max": 159.0
    },
    "lOBlank": {
      "count": 1109.0,
      "mean": 6.745716862037872,
      "std": 12.301209625996982,
      "min": 0.0,
      "25%": 1.0,
      "50%": 2.0,
      "75%": 8.0,
      "max": 225.0
    },
    "locCodeAndComment": {
      "count": 1109.0,
      "mean": 0.9449954914337241,
      "std": 3.3452974598299923,
      "min": 0.0,
      "25%": 0.0,
      "50%": 0.0,
      "75%": 1.0,
      "max": 48.0
    },
    "uniq_Op": {
      "count": 1109.0,
      "mean": 13.307664562669071,
      "std": 8.182224330344004,
      "min": 1.0,
      "25%": 8.0,
      "50%": 12.0,
      "75%": 17.0,
      "max": 99.0
    },
    "uniq_Opnd": {
      "count": 1109.0,
      "mean": 20.89287646528404,
      "std": 29.051434434658468,
      "min": 0.0,
      "25%": 6.0,
      "50%": 12.0,
      "75%": 25.0,
      "max": 538.0
    },
    "total_Op": {
      "count": 1109.0,
      "mean": 66.49341749323715,
      "std": 111.70323564581024,
      "min": 1.0,
      "25%": 15.0,
      "50%": 33.0,
      "75%": 72.0,
      "max": 1641.0
    },
    "total_Opnd": {
      "count": 1109.0,
      "mean": 50.90189359783589,
      "std": 86.30868937153426,
      "min": 0.0,
      "25%": 10.0,
      "50%": 24.0,
      "75%": 56.0,
      "max": 1144.0
    },
    "branchCount": {
      "count": 1109.0,
      "mean": 9.576555455365193,
      "std": 16.540675998659637,
      "min": 1.0,
      "25%": 1.0,
      "50%": 5.0,
      "75%": 11.0,
      "max": 236.0
    }
  },
  "constant_or_near_constant": [],
  "high_correlations": [
    {
      "a": "loc",
      "b": "lOCode",
      "correlation": 0.996508702102699
    },
    {
      "a": "v(g)",
      "b": "branchCount",
      "correlation": 0.9915709990013867
    },
    {
      "a": "n",
      "b": "v",
      "correlation": 0.98727281531597
    },
    {
      "a": "n",
      "b": "b",
      "correlation": 0.9820303694623913
    },
    {
      "a": "n",
      "b": "total_Op",
      "correlation": 0.9973629423685509
    },
    {
      "a": "n",
      "b": "total_Opnd",
      "correlation": 0.9955788273265326
    },
    {
      "a": "v",
      "b": "b",
      "correlation": 0.9952207965573796
    },
    {
      "a": "v",
      "b": "total_Op",
      "correlation": 0.9902703675526342
    },
    {
      "a": "v",
      "b": "total_Opnd",
      "correlation": 0.9756614799573594
    },
    {
      "a": "e",
      "b": "t",
      "correlation": 0.999999999987919
    },
    {
      "a": "b",
      "b": "total_Op",
      "correlation": 0.9850289486944301
    },
    {
      "a": "b",
      "b": "total_Opnd",
      "correlation": 0.970508975981551
    },
    {
      "a": "total_Op",
      "b": "total_Opnd",
      "correlation": 0.9861365601401932
    }
  ],
  "invalid_numeric_values": {
    "loc": 0,
    "v(g)": 0,
    "ev(g)": 0,
    "iv(g)": 0,
    "n": 0,
    "v": 0,
    "l": 0,
    "d": 0,
    "i": 0,
    "e": 0,
    "b": 0,
    "t": 0,
    "lOCode": 0,
    "lOComment": 0,
    "locCodeAndComment": 0,
    "lOBlank": 0,
    "uniq_Op": 0,
    "uniq_Opnd": 0,
    "total_Op": 0,
    "total_Opnd": 0,
    "branchCount": 0
  },
  "infinite_values": {
    "loc": 0,
    "v(g)": 0,
    "ev(g)": 0,
    "iv(g)": 0,
    "n": 0,
    "v": 0,
    "l": 0,
    "d": 0,
    "i": 0,
    "e": 0,
    "b": 0,
    "t": 0,
    "lOCode": 0,
    "lOComment": 0,
    "locCodeAndComment": 0,
    "lOBlank": 0,
    "uniq_Op": 0,
    "uniq_Opnd": 0,
    "total_Op": 0,
    "total_Opnd": 0,
    "branchCount": 0
  },
  "negative_values": {
    "loc": 0,
    "v(g)": 0,
    "ev(g)": 0,
    "iv(g)": 0,
    "n": 0,
    "v": 0,
    "l": 0,
    "d": 0,
    "i": 0,
    "e": 0,
    "b": 0,
    "t": 0,
    "lOCode": 0,
    "lOComment": 0,
    "lOBlank": 0,
    "locCodeAndComment": 0,
    "uniq_Op": 0,
    "uniq_Opnd": 0,
    "total_Op": 0,
    "total_Opnd": 0,
    "branchCount": 0
  },
  "extreme_outliers_3_iqr": {
    "loc": 46,
    "v(g)": 41,
    "ev(g)": 50,
    "iv(g)": 53,
    "n": 58,
    "v": 70,
    "l": 55,
    "d": 14,
    "i": 23,
    "e": 122,
    "b": 69,
    "t": 122,
    "lOCode": 59,
    "lOComment": 64,
    "lOBlank": 47,
    "locCodeAndComment": 54,
    "uniq_Op": 6,
    "uniq_Opnd": 27,
    "total_Op": 52,
    "total_Opnd": 50,
    "branchCount": 34
  }
}
```

## combined

```json
{
  "rows": 17442,
  "unique_feature_groups": 11690,
  "duplicate_feature_groups": 2532,
  "duplicate_feature_rows_beyond_first": 5752,
  "conflicting_label_groups": 121,
  "cross_project_duplicate_groups": 66,
  "cross_project_rows": 1701
}
```

