kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
analysis_df["원래 묶음"] = kmeans.fit_predict(scaled_values)

# 누적 관객 평균이 높은 묶음부터 ㉮, ㉯, ㉰ 부여
cluster_order = (
    analysis_df.groupby("원래 묶음")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)
label_map = {cluster_id: ["㉮", "㉯", "㉰"][i] for i, cluster_id in enumerate(cluster_order)}
analysis_df["영화 유형"] = analysis_df["원래 묶음"].map(label_map)

st.caption("㉮·㉯·㉰는 누적 관객 평균이 높은 유형부터 붙인 이름입니다. 속성 선택에 따라 분류 결과가 달라질 수 있습니다.")

# 시각화용 표시 이름
feature_labels = {
    "스크린 수(로그)": "스크린 수 (상용로그)",
    "누적 관객(로그)": "누적 관객 (상용로그)",
    "10위권 일수": "10위권 일수",
    "롱런 지수": "롱런 지수"
}

st.subheader("2. 2차원 산점도")
col_x, col_y = st.columns(2)
with col_x:
    x_feature = st.selectbox("가로축 속성", options=selected_features, index=0, key="scatter_x")
with col_y:
    y_index = 1 if len(selected_features) > 1 else 0
    y_feature = st.selectbox("세로축 속성", options=selected_features, index=y_index, key="scatter_y")

plot_df = analysis_df.copy()
plot_df["유형"] = plot_df["영화 유형"]
fig2d = px.scatter(
    plot_df,
    x=x_feature,
    y=y_feature,
    color="유형",
    hover_name="movieNm",
    hover_data={"유형": True, x_feature: ":.3f", y_feature: ":.3f"},
    labels={
        x_feature: feature_labels.get(x_feature, x_feature),
        y_feature: feature_labels.get(y_feature, y_feature),
        "유형": "영화 유형"
    },
    title="영화 유형별 2차원 분포"
)
fig2d.update_traces(marker={"size": 7})
fig2d.update_layout(legend_title_text="영화 유형")
st.plotly_chart(fig2d, use_container_width=True)

st.subheader("3. 3차원 산점도")
if len(selected_features) < 3:
    st.info("3차원 산점도를 보려면 분류 속성을 세 개 이상 선택해 주세요.")
else:
    c1, c2, c3 = st.columns(3)
    with c1:
        x3 = st.selectbox("X축 속성", options=selected_features, index=0, key="x3")
    with c2:
        y3 = st.selectbox("Y축 속성", options=selected_features, index=1, key="y3")
    with c3:
        z3 = st.selectbox("Z축 속성", options=selected_features, index=2, key="z3")

    fig3d = px.scatter_3d(
        plot_df,
        x=x3,
        y=y3,
        z=z3,
        color="유형",
        hover_name="movieNm",
        hover_data={"유형": True},
        labels={
            x3: feature_labels.get(x3, x3),
            y3: feature_labels.get(y3, y3),
            z3: feature_labels.get(z3, z3),
            "유형": "영화 유형"
        },
        title="영화 유형별 3차원 분포"
    )
    fig3d.update_traces(marker={"size": 3})
    fig3d.update_layout(legend_title_text="영화 유형")
    st.plotly_chart(fig3d, use_container_width=True)
    st.caption("그래프 안에서 마우스로 드래그하면 시점을 돌려 볼 수 있습니다.")

st.subheader("4. 유형별 편수와 속성 평균")
summary = (
    analysis_df.groupby("영화 유형")
    .agg(
        편수=("movieNm", "size"),
        스크린_수_평균=("first_scrn", "mean"),
        누적_관객_평균=("total_audi", "mean"),
        십위권_일수_평균=("days_in_top10", "mean"),
        롱런_지수_평균=("롱런 지수", "mean")
    )
    .reindex(["㉮", "㉯", "㉰"])
    .reset_index()
)
summary = summary.rename(columns={
    "영화 유형": "영화 유형",
    "스크린_수_평균": "스크린 수 평균",
    "누적_관객_평균": "누적 관객 평균",
    "십위권_일수_평균": "10위권 일수 평균",
    "롱런_지수_평균": "롱런 지수 평균"
})
summary["편수"] = summary["편수"].astype(int)
for col in ["스크린 수 평균", "누적 관객 평균", "10위권 일수 평균", "롱런 지수 평균"]:
    summary[col] = summary[col].round(2)
st.dataframe(summary, use_container_width=True, hide_index=True)

st.subheader("5. 유형별 누적 관객 상위 5편")
for label in ["㉮", "㉯", "㉰"]:
    group = analysis_df[analysis_df["영화 유형"] == label].sort_values("total_audi", ascending=False).head(5)
    with st.expander(f"{label} 유형 — {len(analysis_df[analysis_df['영화 유형'] == label]):,}편 중 상위 {len(group)}편", expanded=True):
        if group.empty:
            st.write("이 유형에 해당하는 영화가 없습니다.")
        else:
            display = group[["movieNm", "total_audi"]].copy()
            display.columns = ["영화 제목", "누적 관객"]
            display["누적 관객"] = display["누적 관객"].map(lambda x: f"{x:,.0f}")
            display.insert(0, "순위", range(1, len(display) + 1))
            st.dataframe(display, use_container_width=True, hide_index=True)

st.caption("참고: 스크린 수와 누적 관객의 로그 값은 분류·산점도용이며, 유형별 평균 표에는 원래 단위의 스크린 수와 누적 관객, 10위권 일수, 롱런 지수를 표시합니다.")
