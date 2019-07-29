<?php
include_once("colors.inc.php");
include_once("colors.names.php");

function hexdiff(&$value,$key,$hex2) {
	$hex1 = $value;
	$r1 = hexdec(substr($hex1,0,2));
	$g1 = hexdec(substr($hex1,2,2));
	$b1 = hexdec(substr($hex1,4,2));
	$r2 = hexdec(substr($hex2,0,2));
	$g2 = hexdec(substr($hex2,2,2));
	$b2 = hexdec(substr($hex2,4,2));
	$value = pow($r1-$r2,2) + pow($g1-$g2,2) + pow($b1-$b2,2);
}

function findcolor($fn,$colorde,$coloren) {
	$ex=new GetMostCommonColors();
	$colors=$ex->Get_Color($fn, $num_results=15, $reduce_brightness=false, $reduce_gradients=true, $delta=16);
    $found=[];
	foreach ($colors as $key => $j) {
        $keep1=$colorde;
		array_walk($keep1,"hexdiff",$key);
		asort($keep1);
		reset($keep1);
		$thiskey1=key($keep1);

        $keep2=$coloren;
		array_walk($keep2,"hexdiff",$key);
		asort($keep2);
		reset($keep2);
		$thiskey2=key($keep2);

		if (!in_array($thiskey1,$found) ){
			echo '<div style="clear:both;height:20px;background-color:#'.$key.';width:'.intval($j*750).'px"></div><div>'.$thiskey1 . " => " . $thiskey2 . '</div>';
			$found[]=$thiskey1;
		}
	}
	echo PHP_EOL;
}

function filterany($element) {
  if ( substr($element,0,1)=="." ) return;
  if ( substr($element,0,6)=="liquid" ) return;
  $bad_words = array('index.html');
  if(in_array($element, $bad_words)) return;
  return $element;
}

$d=realpath(dirname(__FILE__)).'/';
$e='dir/.../';
$u='http://...';

$dh  = opendir($d.'../'.$e.'cache/');

while (false !== ($filename = readdir($dh))) { $files3[] = $filename; }
$files3 = array_filter($files3, "filterany");
#sort($files3);

foreach ($files3 as $i) {
	$image_url=$u.$e.'/'.$i;
	print <<<EOF
<div style="float:left">
<i style="margin-top:50px;"></i>
<img src="{$u}{$e}cache/{$i}">

EOF;
findcolor('../'.$e.'cache/'.$i,$colorde,$coloren);
echo "</div>";

}
?>
