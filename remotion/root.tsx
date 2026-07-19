/** Composition metadata and manifest props boundary. */
import React from 'react';
import {CalculateMetadataFunction, Composition} from 'remotion';
import {FinancialVideo, Manifest} from './video';

const fallback: Manifest={output:{width:1920,height:1080,fps:30},scenes:[],narration:{segments:[]},charts:{charts:[]},timeline:{total_frames:2100,events:[]},disclaimer:''};

/** Derive duration and dimensions from each approved render manifest. */
const calculateMetadata:CalculateMetadataFunction<Manifest>=({props})=>({durationInFrames:props.timeline.total_frames||2100,fps:props.output.fps||30,width:props.output.width||1920,height:props.output.height||1080});

/** Registers a fixed renderer contract; JSON props supply content, never executable animation code. */
export const Root:React.FC=()=> <Composition id="FinancialVideo" component={FinancialVideo} durationInFrames={2100} fps={30} width={1920} height={1080} defaultProps={fallback} calculateMetadata={calculateMetadata}/>;
